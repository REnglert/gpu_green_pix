#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"

#define DEFAULT_THRESHOLD 4000

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){
  
    if ( !filename || filename[0] == '\0') {
        fprintf(stderr, "read_ppm but no file name\n");
        return NULL;  // fail
    }

    fprintf(stderr, "read_ppm( %s )\n", filename);
    int fd = open( filename, O_RDONLY);
    if (fd == -1){
        fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
        return NULL; // fail 
    }

    char chars[1024];
    int num = read(fd, chars, 1000);

      if (chars[0] != 'P' || chars[1] != '6'){
        fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
        return NULL;
    }

     unsigned int width, height, maxvalue;


    char *ptr = chars+3; // P 6 newline
    if (*ptr == '#'){ // comment line! 
        ptr = 1 + strstr(ptr, "\n");
    }

    num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
    fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
    xsize = width;
    ysize = height;
    maxval = maxvalue;
  
    unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int) * 3);
    if (!pic) {
        fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
        return NULL; // fail but return
    }

  // allocate buffer to read the rest of the file into
    int bufsize =  3 * width * height * sizeof(unsigned char);
    if (maxval > 255) bufsize *= 2;
    unsigned char *buf = (unsigned char *)malloc( bufsize );
    if (!buf) {
        fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
        return NULL; // fail but return
    }

    // TODO really read
    char duh[80];
    char *line = chars;

    // find the start of the pixel data.   no doubt stupid
    sprintf(duh, "%d\0", xsize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", ysize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", maxval);
    line = strstr(line, duh);


    fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
    line += strlen(duh) + 1;

    long offset = line - chars;
    lseek(fd, offset, SEEK_SET); // move to the correct offset
    long numread = read(fd, buf, bufsize);
    fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

    close(fd);

    int pixels = xsize * ysize * 3;
    for (int i=0; i<pixels; i++) pic[i] = (int) buf[i]; 

    return pic; // success
}

void write_ppm( const char *filename, int xsize, int ysize, int maxval, int *pic) 
{
    FILE *fp;
    
    fp = fopen(filename, "w");
    if (!fp) {
            fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
            exit(-1); 
    }
    
    fprintf(fp, "P6\n"); 
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
    
    int numpix = xsize * ysize * 3;
    for (int i=0; i<numpix; i+=3) {
        fprintf(fp, "%c%c%c", (unsigned char) pic[i], (unsigned char) pic[i+1], (unsigned char) pic[i+2]); 
    }
    fclose(fp);
}

__global__ void gcount_baseline(unsigned int *pic, int *result, int xsize, int ysize, int *count){
    int j = 3*(blockIdx.x*blockDim.x + threadIdx.x); // col
    int i = blockIdx.y*blockDim.y + threadIdx.y; // row
    int offset = i*xsize*3 + j; // location of red value

    if( j < xsize*3 || i < ysize){
        int r = pic[offset];
        int g = pic[offset+1];
        int b = pic[offset+2];

        if(g > r && g > b){
            r = 255;
            g = 0;
            b = 0;
            atomicAdd(count, 1);
        }

        result[offset] = r;
        result[offset+1] = g;
        result[offset+2] = b;
    }
}

void checkCudaError(const char* task){
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Oops! (error code %s happened at \"%s\")!\n", cudaGetErrorString(err), task); 
        exit(EXIT_FAILURE);
    }

    // fprintf(stderr, "Success! Completed \"%s\"!\n", task);
}

main( int argc, char **argv ){

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
    filename = "img/img_1.ppm";
  
    if (argc > 1) {
        if (argc == 3)  { // filename AND threshold
            filename = strdup( argv[1]);
            thresh = atoi( argv[2] );
        }
        if (argc == 2) { // default file but specified threshhold
        
            thresh = atoi( argv[1] );
        }

        fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
    }


    int xsize, ysize, maxval;
    unsigned int *pic = read_ppm( filename, xsize, ysize, maxval ); 

    int numbytes =  xsize * ysize * 3 * sizeof( int );

    cudaEvent_t start_event, stop_event;
    float elapsed_time_par;

    int *result = (int *) malloc( numbytes );
    unsigned int *d_pic = NULL;
    int *d_result = NULL;

    cudaMalloc((void **) &d_pic, numbytes);
    checkCudaError("allocate d_pic");

    cudaMalloc((void **) &d_result, numbytes);
    checkCudaError("allocate d_result");

    cudaMemcpy(d_pic, pic, xsize * ysize * sizeof(unsigned int) * 3 , cudaMemcpyHostToDevice);
    checkCudaError("copy d_pic");

    // Launch the CUDA Kernel
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(ceil(xsize/ (float)BLOCK_SIZE_X ), ceil(ysize/ (float)BLOCK_SIZE_Y ));

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    int *num_pix_found;
    cudaMallocManaged(&num_pix_found, 4);
    *num_pix_found = 0;

    // Launch kernel function
    gcount_baseline<<<grid, block>>>(d_pic, d_result, xsize, ysize, num_pix_found); 
    checkCudaError("kernel launch");

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "Parallel Runtime: %f ms\n", elapsed_time_par);

    cudaMemcpy(result, d_result, numbytes, cudaMemcpyDeviceToHost);
    checkCudaError("copy d_result");

    write_ppm( "result.ppm", xsize, ysize, 255, result);

    fprintf(stderr, "num_pix_found = %d\n", *num_pix_found);
}
