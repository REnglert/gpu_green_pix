#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"

#define DEFAULT_THRESHOLD 4000

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

typedef struct {
    unsigned int r,g,b;
} Pixels;

Pixels *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){
    
    if ( !filename || filename[0] == '\0') {
        fprintf(stderr, "read_ppm but no file name\n");
        return NULL;  // fail
    }
    
    // fprintf(stderr, "read_ppm( %s )\n", filename);
    int fd = open( filename, O_RDONLY);
    if (fd == -1)
    {
        fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
        return NULL; // fail
        
    }
    
    char chars[1024];
    int num = read(fd, chars, 1000);
    
    if (chars[0] != 'P' || chars[1] != '6')
    {
        fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
        return NULL;
    }
    
    unsigned int width, height, maxvalue;
    
    
    char *ptr = chars+3; // P 6 newline
    if (*ptr == '#') // comment line!
    {
        ptr = 1 + strstr(ptr, "\n");
    }
    
    num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
    // fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);
    xsize = width;
    ysize = height;
    maxval = maxvalue;
    
    Pixels *pic = (Pixels *)malloc( width * height * sizeof(Pixels));
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
    
    
    int pixels = xsize * ysize*3;
    int pos = 0; 
    for (int i=0; i<pixels; i+=3){
         pic[pos].r = (int) buf[i];  // red channel
         pic[pos].g = (int) buf[i+1];  // green channel
         pic[pos].b = (int) buf[i+2];  // blue channel
         pos++; 
    }

    return pic; // success
    
}


void write_ppm( char *filename, int xsize, int ysize, int maxval, Pixels *pic) 
{
    FILE *fp;
    
    fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
        exit(-1);
    }
    //int x,y;
    
    
    fprintf(fp, "P6\n");
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
    
    int numpix = xsize * ysize;
    for (int i=0; i<numpix; i++) {
        fprintf(fp, "%c%c%c", (unsigned char) pic[i].r, (unsigned char) pic[i].g, (unsigned char) pic[i].b);
    }
    fclose(fp);
    
}

void write_ppm1D( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
    FILE *fp;
    
    fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
        exit(-1);
    }
    //int x,y;
    
    
    fprintf(fp, "P6\n");
    fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
    
    int numpix = xsize * ysize;
    for (int i=0; i<numpix; i++) {
        unsigned char uc = (unsigned char) pic[i];
        fprintf(fp, "%c%c%c", uc, uc, uc);
    }
    fclose(fp);
    
}

__global__ void gauss_filter(Pixels *pic, Pixels *result, int xsize, int ysize, double *blur, int mask_size){
    int j = blockIdx.x * blockDim.x + threadIdx.x; //col 
    int i = blockIdx.y * blockDim.y + threadIdx.y; //row 
    int offset = i*xsize + j; 

    if(i >= 0 && j >= 0 && i < ysize && j < xsize){        
        double sum = 0; 
        double redVal = 0; 
        double greenVal = 0; 
        double blueVal = 0; 

        for (int i = 0; i < mask_size; ++i) {
            for (int j = 0; j < mask_size; ++j) {    
                if (((offset + ((i - ((mask_size - 1) / 2))*xsize) + j - ((mask_size - 1) / 2)) >= 0)
                    && ((offset + ((i - ((mask_size - 1) / 2))*xsize) + j - ((mask_size - 1) / 2)) <= ysize*xsize-1)
                    && (((offset % xsize) + j - ((mask_size-1)/2)) >= 0)
                    && (((offset % xsize) + j - ((mask_size-1)/2)) <= (xsize-1))) {

                    redVal += blur[i * mask_size + j] * pic[offset + ((i - ((mask_size - 1) / 2))*xsize) + j - ((mask_size - 1) / 2)].r;
                    greenVal += blur[i * mask_size + j] * pic[offset + ((i - ((mask_size - 1) / 2))*xsize) + j - ((mask_size - 1) / 2)].g;
                    blueVal += blur[i * mask_size + j] * pic[offset + ((i - ((mask_size - 1) / 2))*xsize) + j - ((mask_size - 1) / 2)].b;
                    sum += blur[i * mask_size + j];
                }
            }
        }
        
        //update output image
        result[offset].r = (redVal / sum);
        result[offset].g = (greenVal / sum);
        result[offset].b = (blueVal / sum);
    }
}


__global__ void laplacian(Pixels *pic, int *d_result_l, bool *isEdge, int xsize, int ysize){
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    // Ignore any small edges detected
    int thresh = 550;
 
    int w = 7;
    int h = 7;

    //float kernel[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    //float kernel[5][5] = {0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0}; 
    float kernel[7][7] = {0,0,-1,-1,-1,0,0,0,-1,-3,-3,-3,-1,0, 
				-1,-3,0,7,0,-3,-1,-1,-3,7,24,7,-3,-1,
				-1,-3,0,7,0,-3,-1,0,-1,-3,-3,-3,-1,0,0,0,-1,-1,-1,0,0};

    if((j >= w/2) && (j < (xsize - w/2)) && (i >= h/2) && (i < (ysize - h/2))){
    	int sum = 0;
	for(int y = -h/2; y <= h/2; y++){
	   for(int x = -w/2; x < w/2; x++){
		Pixels p = pic[((i+y)*xsize + (j+x))];
		int f = (p.r*0.299) + (p.g*0.587) + (p.b*0.114);
		sum += f * kernel[y + h/2][x + w/2];
	   }
	}
	int offset = i*xsize + j;
	//d_result_l[offset] = sum;
 
	if(sum > thresh){
	   d_result_l[offset] = sum;
	   isEdge[offset] = true;
	}
	else {
	   d_result_l[offset] = 0;
	   isEdge[offset] = false;
	}
    }
}


__global__ void gcount_perpixel(Pixels *pic, bool *isEdge, Pixels *result, int xsize, int ysize, int *count, int startX, int startY){
    
    int col = startX + threadIdx.x;
    int row = startY + threadIdx.y;

    int offset = row*xsize + col; // location of green value

    if( col < xsize && row < ysize){

        int r = pic[offset].r;
        int g = pic[offset].g;
        int b = pic[offset].b;

        int thresh = 10;
        if(g-thresh > r && g-thresh > b){
            atomicAdd(count, 1);
            r = 140;
            b=g=0;
        }
        

        if(isEdge[offset]){
            result[offset].r = 255;
            result[offset].g = 255;
            result[offset].b = 255;
        } 
        else {
            result[offset].r = r;
            result[offset].g = g;
            result[offset].b = b;
        }  

    }  

}

__global__ void gcount(Pixels *pic, bool *isEdge, Pixels *result, int xsize, int ysize, int *count){

    int cols = 16;
    int rows = 16;

    //TODO: I could speed this up by transposing the matrix and skipping unneeded rows

    int col = cols*(blockIdx.x*blockDim.x + threadIdx.x); // col
    int row = rows*(blockIdx.y*blockDim.y + threadIdx.y); // row

    // get average color
    float r = 0;
    float g = 0;
    float b = 0;

    int edgeCount = 0;

    for(int i = 0; i < cols; i++){
        for(int j = 0; j < rows; j++){
            int offset = (row+j)*xsize + (col+i);

            Pixels p = pic[offset];
    
            if( col < xsize && row < ysize){
                r += p.r;
                g += p.g;
                b += p.b;

                edgeCount += (int)isEdge[offset];

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
                // int offset = (row+j)*xsize*3 + (col+i*3); // location of red value
                int offset = (row+j)*xsize + (col+i);
        
                if( col < xsize && row < ysize){
    
                    if(isEdge[offset]){
                        result[offset].r = 255;
                        result[offset].g = 255;
                        result[offset].b = 255;
                    } 
                    else {
                        result[offset].r = (int)r;
                        result[offset].g = (int)g;
                        result[offset].b = (int)b;
                    }  
                }
            }
        }
    }

}

double * gaussian_mask(int mask_size=7, double pi=3.14, double scaled =1){
    double *gauss_mask = (double *)malloc(mask_size * mask_size * sizeof(double)); 
    double sigma = (double) mask_size / 3; 

    for(int i = 0; i < mask_size; i++){
        for(int j = 0; j < mask_size; j++){
            double iComponent = pow((i-mask_size)/2, 2); 
            double jComponent = pow((j-mask_size)/2, 2); 

            double sig2 = pow(sigma, 2); 
            double normal = exp(-(((iComponent) + (jComponent)) / (2 * sig2)));
            double gVal = (1 / (sqrt(2 * pi)*sigma)) * normal; 
            int offset = i * mask_size + j; 

            gauss_mask[offset] = gVal; 

            if(i == 0 && j==0){
                scaled = gauss_mask[0]; 
            }

            gauss_mask[offset] = gauss_mask[offset] / scaled; 
        }
    }

    return gauss_mask; 
}

void checkCudaError(const char* task){
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Oops! (error code %s happened at \"%s\")!\n", cudaGetErrorString(err), task); 
        exit(EXIT_FAILURE);
    }
}

main( int argc, char **argv ){

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
    filename = "img/img_4.ppm";
  
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
    Pixels * pic = read_ppm( filename, xsize, ysize, maxval ); 
    
    int numbytes =  xsize * ysize * sizeof( Pixels );
    int numbools =  xsize * ysize * sizeof( bool ); // edge detection boolean size    

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(ceil(xsize/ (float)BLOCK_SIZE_X ), ceil(ysize/ (float)BLOCK_SIZE_Y ));    

    //Step 1: Gaussian Filter (Blur)
    //Compute Gaussian mask
    int mask_size = 7;
    double *gauss_mask = gaussian_mask();
 
    cudaEvent_t start_event, stop_event;  
    float elapsed_time_par;

    Pixels *result = (Pixels *) malloc( numbytes ); // host and device result image array
    Pixels *d_result = NULL;
    
    bool *isEdge = (bool *) malloc( numbools ); // host and device edge boolean array
    bool *d_isEdge = NULL;

    Pixels *blur_result = (Pixels *) malloc( numbytes );
    int *laplac_result = (int *) malloc( xsize*ysize*sizeof(int) );
    int *d_result_l = NULL;
    Pixels *d_pic = NULL;
    Pixels *d_result_b = NULL; 
    double *d_blur = NULL; 

    //========Sequential Laplacian==================
    Pixels *stage1 = (Pixels *)malloc(numbytes); 
    int* stage2 = (int *)malloc(xsize * ysize * sizeof(int)); 
    bool* sEdge = (bool *)malloc(numbools);

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    //==Gaussian==
    for(int i = 0; i < ysize; i++){
        for(int j = 0; j < xsize; j++){
            int offset = i*xsize+j; 
            double sum = 0; 
            double redVal = 0; 
            double greenVal = 0; 
            double blueVal = 0; 
    
            for (int a = 0; a < mask_size; ++a) {
                for (int b = 0; b < mask_size; ++b) {    
                    if (((offset + ((a - ((mask_size - 1) / 2))*xsize) + b - ((mask_size - 1) / 2)) >= 0)
                        && ((offset + ((a - ((mask_size - 1) / 2))*xsize) + b - ((mask_size - 1) / 2)) <= ysize*xsize-1)
                        && (((offset % xsize) + b - ((mask_size-1)/2)) >= 0)
                        && (((offset % xsize) + b - ((mask_size-1)/2)) <= (xsize-1))) {
    
                        redVal += gauss_mask[a * mask_size + b] * pic[offset + ((a - ((mask_size - 1) / 2))*xsize) + b - ((mask_size - 1) / 2)].r;
                        greenVal += gauss_mask[a * mask_size + b] * pic[offset + ((a - ((mask_size - 1) / 2))*xsize) + b - ((mask_size - 1) / 2)].g;
                        blueVal += gauss_mask[a * mask_size + b] * pic[offset + ((a - ((mask_size - 1) / 2))*xsize) + b - ((mask_size - 1) / 2)].b;
                        sum += gauss_mask[a * mask_size + b];
                    }
                }
            }
            
            //update output image
            stage1[offset].r = (redVal / sum);
            stage1[offset].g = (greenVal / sum);
            stage1[offset].b = (blueVal / sum);
        }
    }


    //==Laplacian==
    float kernel[7][7] = {0,0,-1,-1,-1,0,0,0,-1,-3,-3,-3,-1,0, 
				-1,-3,0,7,0,-3,-1,-1,-3,7,24,7,-3,-1,
				-1,-3,0,7,0,-3,-1,0,-1,-3,-3,-3,-1,0,0,0,-1,-1,-1,0,0};
    int w = 7;
    int h = 7;
    int t = 550;
    for(int i = 0; i < ysize; i++){
        for(int j = 0; j < xsize; j++){
	   if((j >= w/2) && (j < (xsize - w/2)) && (i >= h/2) && (i < (ysize - h/2))){
    	      int sum = 0;
	      for(int y = -h/2; y <= h/2; y++){
	         for(int x = -w/2; x < w/2; x++){
		      Pixels p = stage1[((i+y)*xsize + (j+x))];
		      int f = (p.r*0.299) + (p.g*0.587) + (p.b*0.114);
		      sum += f * kernel[y + h/2][x + w/2];
	         }
	      }
	      int offset = i*xsize + j;
 
  	      if(sum > t){
	         stage2[offset] = sum;
	         sEdge[offset] = true;
	      }
	      else {
	         stage2[offset] = 0;
	         sEdge[offset] = false;
	      }
	   }	
	}
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Edge Detection Sequential Runtime: %f ms\n", elapsed_time_par);

    free(stage1);
    free(stage2);
    free(sEdge);

    cudaMalloc((void**) &d_blur, sizeof(d_blur[0])*mask_size*mask_size);
    checkCudaError("malloc d_blur");
    cudaMalloc((void **) &d_pic, numbytes);
    checkCudaError("malloc d_pic");
    cudaMalloc((void **) &d_result_b, numbytes);
    checkCudaError("malloc d_result_b");
    cudaMalloc((void **) &d_result_l, xsize*ysize*sizeof(int) );
    checkCudaError("malloc d_result_l");

    cudaMemcpy(d_blur, gauss_mask, sizeof(d_blur[0])*mask_size*mask_size, cudaMemcpyHostToDevice);
    checkCudaError("copy gauss mask ");
    cudaMemcpy(d_pic, pic, numbytes, cudaMemcpyHostToDevice);
    checkCudaError("copy pic");
    cudaMalloc((void **) &d_isEdge, numbools); // allocate isEdge space on device
    checkCudaError("allocate d_isEdge"); 
    int* d_gradient_result = NULL; 
    cudaMalloc((void **) &d_gradient_result, xsize*ysize*sizeof(int) );
    checkCudaError("malloc d_gradient_result");

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    // Launch blur function
    gauss_filter<<<grid, block>>>(d_pic, d_result_b, xsize, ysize, d_blur, mask_size); 
    

    // Step 2: laplacian filter
    laplacian<<<grid, block>>>(d_result_b, d_result_l, d_isEdge, xsize, ysize);
   
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Edge Detection Parallel Runtime: %f ms\n", elapsed_time_par);

    cudaMemcpy(blur_result, d_result_b, numbytes, cudaMemcpyDeviceToHost);
    checkCudaError("copy d_result_b");

    write_ppm( "blur.ppm", xsize, ysize, 255, blur_result);

    //cudaMemcpy(gradient_result, d_gradient_result, xsize*ysize*sizeof(int), cudaMemcpyDeviceToHost);
    //checkCudaError("copy gradient result");

    //write_ppm1D( "gradient.ppm", xsize, ysize, 255, gradient_result);



    cudaMemcpy(laplac_result, d_result_l, xsize*ysize*sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("copy d_result_l");

    write_ppm1D( "laplacian.ppm", xsize, ysize, 255, laplac_result);


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
    gcount<<<grid, block>>>(d_pic, d_isEdge, d_result, xsize, ysize, num_pix_found); 
    checkCudaError("kernel launch");

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Pixel Counting Parallel Runtime: %f ms\n", elapsed_time_par);

    cudaMemcpy(result, d_result, numbytes, cudaMemcpyDeviceToHost); // copy result image to host
    checkCudaError("copy d_result");

    write_ppm( "laplacian.ppm", xsize, ysize, 255, result); // write result image file

    fprintf(stderr, "   file: %s, num_pix_found: %d, cm^2: %d\n",filename, *num_pix_found, *num_pix_found / 467); // there are 466.667 pixels per cm^2

    // fprintf(stderr, "Finished!!!!!!\n");
}
