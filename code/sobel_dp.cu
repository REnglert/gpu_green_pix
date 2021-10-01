#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"
#include "sobel.h"

#define DEFAULT_THRESHOLD 12000

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){
  
    if ( !filename || filename[0] == '\0') {
        fprintf(stderr, "read_ppm but no file name\n");
        return NULL;  // fail
    }

    // fprintf(stderr, "read_ppm( %s )\n", filename);
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
    // fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
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


    // fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
    line += strlen(duh) + 1;

    long offset = line - chars;
    lseek(fd, offset, SEEK_SET); // move to the correct offset
    long numread = read(fd, buf, bufsize);
    // fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

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

void write_ppm_from_bools( const char *filename, int xsize, int ysize, int maxval, bool *pic) 
{
    FILE *fp;
    
    fp = fopen(filename, "w");
    if (!fp) {
            fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
            exit(-1); 
    }
    
    fprintf(fp, "P6\n"); 
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
    
    int numpix = xsize * ysize;
    for (int i=0; i<numpix; i++) {
        int val = 0;
        if(pic[i]){ val = 255; }
        fprintf(fp, "%c%c%c", (unsigned char) val, (unsigned char) val, (unsigned char) val); 
    }
    fclose(fp);
}

__global__ void sobel(unsigned int *pic, bool *isEdge, int xsize, int ysize){
    int j = 3*(blockIdx.x*blockDim.x + threadIdx.x) + 1; // col
    int i = blockIdx.y*blockDim.y + threadIdx.y; // row

    int sum1, sum2, magnitude;
    int thresh = DEFAULT_THRESHOLD;

    if(i >= 1 && i < ysize-1 && j >= 1 && j < xsize*3-1){
        sum1 = pic[ (i-1)*xsize*3 + j+3 ] - pic[ (i-1)*xsize*3 + j-3 ] 
        + 2 * pic[ (i)*xsize*3 + j+3 ] - 2 * pic[ (i)*xsize*3 + j-3 ]
        + pic[ (i+1)*xsize*3 + j+3 ] - pic[ (i+1)*xsize*3 + j-3 ];
        
        sum2 = pic[ (i-1)*xsize*3 + j-3 ] + 2 * pic[ (i-1)*xsize*3 + j] + pic[ (i-1)*xsize*3 + j+3 ]
        - pic[ (i+1)*xsize*3 + j-3 ] - 2 * pic[ (i+1)*xsize*3 + j ] - pic[ (i+1)*xsize*3 + j+3 ];
        
        magnitude = sum1*sum1 + sum2*sum2;

        int offset = i*xsize + blockIdx.x*blockDim.x + threadIdx.x;
        if (magnitude > thresh){
            isEdge[offset] = true;
        }
        else {
            isEdge[offset] = false;
        } 
    }
}

__global__ void gcount_perpixel(unsigned int *pic, bool *isEdge, int *result, int xsize, int ysize, int *count, int startX, int startY){
    
    int col = startX + threadIdx.x*3;
    int row = startY + threadIdx.y;

    int offset = row*xsize*3 + col; // location of green value

    if( col < xsize*3 && row < ysize){
        int r = pic[offset];
        int g = pic[offset+1];
        int b = pic[offset+2];

        int thresh = 10;
        if(g-thresh > r && g-thresh > b){
            atomicAdd(count, 1);
            r = 140;
            b=g=0;
        }
        

        if(isEdge[row*xsize + col/3]){
            result[offset] = 255;
            result[offset+1] = 255;
            result[offset+2] = 255;
        } 
        else {
            result[offset] = r;
            result[offset+1] = g;
            result[offset+2] = b;
        }  

    }  

}

__global__ void gcount(unsigned int *pic, bool *isEdge, int *result, int xsize, int ysize, int *count){

    int cols = 16;
    int rows = 16;

    //TODO: I could speed this up by transposing the matrix and skipping unneeded rows

    int col = cols*3*(blockIdx.x*blockDim.x + threadIdx.x); // col
    int row = rows*(blockIdx.y*blockDim.y + threadIdx.y); // row

    // get average color
    float r = 0;
    float g = 0;
    float b = 0;

    int edgeCount = 0;

    for(int i = 0; i < cols*3; i+=3){
        for(int j = 0; j < rows; j++){
            int offset = (row+j)*xsize*3 + (col+i); // location of red value
    
            if( col < xsize*3 && row < ysize){
                r += pic[offset];
                g += pic[offset+1];
                b += pic[offset+2];

                edgeCount += (int)isEdge[(row+j)*xsize + (col/3+i)];

            }
        }
    }

    if(edgeCount > 25){

        dim3 grid(1, 1);
        dim3 block(16, 16);

        gcount_perpixel<<<grid, block>>>(pic, isEdge, result, xsize, ysize, count, col, row); 
    } else {
        r = r / (float)(cols*rows);
        g = g / (float)(cols*rows);
        b = b / (float)(cols*rows);
    
        int thresh = 10;
        if(g-thresh > r && g-thresh > b){
            atomicAdd(count, cols*rows);
            r=255;
            b=g=0;
        }

        for(int i = 0; i < cols; i++){
            for(int j = 0; j < rows; j++){
                int offset = (row+j)*xsize*3 + (col+i*3); // location of red value
        
                if( col < xsize*3 && row < ysize){
    
                    if(isEdge[(row+j)*xsize + (col/3+i)]){
                        result[offset] = 255;
                        result[offset+1] = 255;
                        result[offset+2] = 255;
                    } 
                    else {
                        result[offset] = (int)r;
                        result[offset+1] = (int)g;
                        result[offset+2] = (int)b;
                    }  
                }
            }
        }
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

int main(int argc, char *argv[]){

    char* filename = strdup( argv[1] );
    int xsize, ysize, maxval;
    unsigned int *pic = read_ppm( filename, xsize, ysize, maxval); // define variables and read in image file

    // LOADING AND SETUP CODE ===================================================
    int numbytes =  xsize * ysize * 3 * sizeof( int ); // 3x because 3 floats for R, G, B channels
    int numbools =  xsize * ysize * sizeof( bool ); // edge detection boolean size

    cudaEvent_t start_event, stop_event; // 
    float elapsed_time_par;
    
    unsigned int *d_pic = NULL; // pointer for device picture array

    bool *isEdge = (bool *) malloc( numbools ); // host and device edge boolean array
    bool *d_isEdge = NULL;

    int *result = (int *) malloc( numbytes ); // host and device result image array
    int *d_result = NULL;

    // SEQUENTIAL SOBEL ===================================================
    bool *seqIsEdge = (bool *) malloc( numbools );

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    for(int i = 0; i < ysize; i++){
        for (int j = 0; j < xsize; j++){
            int col = j*3 + 1;
            int sum1, sum2, magnitude;
            int thresh = DEFAULT_THRESHOLD;
        
            if(i >= 1 && i < ysize-1 && col >= 1 && col < xsize*3-1){
                sum1 = pic[ (i-1)*xsize*3 + col+3 ] - pic[ (i-1)*xsize*3 + col-3 ] 
                + 2 * pic[ (i)*xsize*3 + col+3 ] - 2 * pic[ (i)*xsize*3 + col-3 ]
                + pic[ (i+1)*xsize*3 + col+3 ] - pic[ (i+1)*xsize*3 + col-3 ];
                
                sum2 = pic[ (i-1)*xsize*3 + col-3 ] + 2 * pic[ (i-1)*xsize*3 + col] + pic[ (i-1)*xsize*3 + col+3 ]
                - pic[ (i+1)*xsize*3 + col-3 ] - 2 * pic[ (i+1)*xsize*3 + col ] - pic[ (i+1)*xsize*3 + col+3 ];
                
                magnitude = sum1*sum1 + sum2*sum2;
        
                int offset = i*xsize + j;
                if (magnitude > thresh){
                    seqIsEdge[offset] = true;
                }
                else {
                    seqIsEdge[offset] = false;
                } 
            }
        }
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Edge Detection Sequential Runtime: %f ms\n", elapsed_time_par);

    cudaMalloc((void **) &d_pic, numbytes); // allocate input image space on device
    checkCudaError("allocate d_pic");

    cudaMemcpy(d_pic, pic, xsize * ysize * sizeof(unsigned int) * 3 , cudaMemcpyHostToDevice); // copy input image to device
    checkCudaError("copy d_pic");

    cudaMalloc((void **) &d_isEdge, numbools); // allocate isEdge space on device
    checkCudaError("allocate d_isEdge");

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(ceil(xsize/ (float)BLOCK_SIZE_X ), ceil(ysize/ (float)BLOCK_SIZE_Y ));

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    // Launch edge detection kernel function
    // takes in pic array, returns boolean isEdge
    sobel<<<grid, block>>>(d_pic, d_isEdge, xsize, ysize); 
    checkCudaError("kernel launch");

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Edge Detection Parallel Runtime: %f ms\n", elapsed_time_par);

    // GREEN PIXEL COUNTING CODE ================================================
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    int count = 0;

    for(int i = 0; i < ysize; i++){
        for (int j = 0; j < xsize; j++){

            //TODO: I could speed this up by transposing the matrix and skipping unneeded rows

            int col = j*3; // col
            int row = i; // row

            // get average color
            float r = 0;
            float g = 0;
            float b = 0;

            for(int i = 0; i < 3; i+=3){
                for(int j = 0; j < 1; j++){
                    int offset = (row+j)*xsize*3 + (col+i); // location of red value
            
                    if( col < xsize*3 && row < ysize){
                        r += pic[offset];
                        g += pic[offset+1];
                        b += pic[offset+2];
                    }
                }
            }

            int thresh = 10;
            if(g-thresh > r && g-thresh > b){
                count++;
                r=255;
                b=g=0;
            }

            for(int i = 0; i < 1; i++){
                for(int j = 0; j < 1; j++){
                    int offset = (row+j)*xsize*3 + (col+i*3); // location of red value
            
                    if( col < xsize*3 && row < ysize){

                        if(isEdge[(row+j)*xsize + (col/3+i)]){
                            result[offset] = 255;
                            result[offset+1] = 255;
                            result[offset+2] = 255;
                        } 
                        else {
                            result[offset] = (int)r;
                            result[offset+1] = (int)g;
                            result[offset+2] = (int)b;
                        }  
                    }
                }
            }
        }
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Pixel Counting Sequential Runtime: %f ms\n", elapsed_time_par);

    cudaMalloc((void **) &d_result, numbytes); // allocate result image space on device
    checkCudaError("allocate d_result");

    dim3 grid2(ceil((xsize/16)/ (float)BLOCK_SIZE_X ), ceil((ysize/16)/ (float)BLOCK_SIZE_Y ));

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    int *num_pix_found;
    cudaMallocManaged(&num_pix_found, 4); // allocate space for num_pix_found on device
    *num_pix_found = 0;

    // Launch pixel count kernel function
    // takes in input pic array and boolean isEdge, returns num_pix_found and result image array
    gcount<<<grid2, block>>>(d_pic, d_isEdge, d_result, xsize, ysize, num_pix_found); 
    checkCudaError("kernel launch");

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Pixel Counting Parallel Runtime: %f ms\n", elapsed_time_par);

    // fprintf(stderr, "Count Runtime: %f ms\n", elapsed_time_par);

    cudaMemcpy(result, d_result, numbytes, cudaMemcpyDeviceToHost); // copy result image to host
    checkCudaError("copy d_result");

    fprintf(stderr, "   file: %s, num_pix_found: %d, cm^2: %d\n",filename, *num_pix_found, *num_pix_found / 467); // there are 466.667 pixels per cm^2

    cudaFree(d_pic);
    cudaFree(d_isEdge);
    cudaFree(d_result);

    write_ppm("sobel.ppm", xsize, ysize, 255, result); // write result image file
}
