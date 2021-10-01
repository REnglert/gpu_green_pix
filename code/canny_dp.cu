#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"

#define DEFAULT_THRESHOLD 4000

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define TS 32 
#define EDGE 1 

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

void checkCudaError(const char* task){
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Oops! (error code %s happened at \"%s\")!\n", cudaGetErrorString(err), task); 
        exit(EXIT_FAILURE);
    }

    // fprintf(stderr, "Success! Completed \"%s\"!\n", task);
}

/**
    Apply a guassion filter. Blur the hard edges. 
*/
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

/**
    delta = f(x-1) * f(x+1)
*/
__global__ void gradient_calc(Pixels *pic, int *result, int *deltaX, int *deltaY, int xsize, int ysize){
    int j = blockIdx.x*blockDim.x + threadIdx.x; // col
    int i = blockIdx.y*blockDim.y + threadIdx.y; // row
    int offset = i*xsize + j; 

    //Skip first/last row 
    if(i > 0 &&  i < (ysize-1)){
        int16_t xRed = 0;
        int16_t yRed = 0;
        int16_t xGreen = 0;
        int16_t yGreen = 0;
        int16_t xBlue = 0;
        int16_t yBlue = 0;
        int dX = 0; 
        int dY = 0; 

        //3 cases 
        //1st col 
        if((j % xsize) == 0){
            xRed = (int16_t)(pic[offset +1].r       - pic[offset].r );
            yRed = (int16_t)(pic[offset + xsize].r  - pic[offset].r ); 

            xGreen = (int16_t)(pic[offset +1].g      - pic[offset].g );
            yGreen = (int16_t)(pic[offset + xsize].g - pic[offset].g ); 
            
            xBlue = (int16_t)(pic[offset +1].b      - pic[offset].b );
            yBlue = (int16_t)(pic[offset + xsize].b - pic[offset].b ); 
        }
        //last col 
        else if((j % xsize) == xsize -1){
            xRed = (int16_t)(pic[offset].r - pic[offset - 1].r ); 
            yRed = (int16_t)(pic[offset].r - pic[offset - xsize].r ); 

            xGreen = (int16_t)(pic[offset].g - pic[offset - 1].g ); 
            yGreen = (int16_t)(pic[offset].g - pic[offset - xsize].g ); 

            xBlue = (int16_t)(pic[offset].b  - pic[offset - 1].b ); 
            yBlue = (int16_t)(pic[offset].b  - pic[offset - xsize].b ); 
        }
        //middle col 
        else {
            xRed = (int16_t)(pic[offset + 1].r - pic[offset - 1].r ); 
            yRed = (int16_t)(pic[offset + xsize].r - pic[offset - xsize].r ); 

            xGreen = (int16_t)(pic[offset + 1].g  - pic[offset - 1].g ); 
            yGreen = (int16_t)(pic[offset + xsize].g  - pic[offset - xsize].g ); 

            xBlue = (int16_t)(pic[offset + 1].b  - pic[offset - 1].b ); 
            yBlue = (int16_t)(pic[offset + xsize].b  - pic[offset - xsize].b ); 
        }

        dX= (int)(0.2989 * xRed + 0.5870 * xGreen + 0.1140 * xBlue);
        dY= (int)(0.2989 * yRed + 0.5870 * yGreen + 0.1140 * yBlue); 

        // deltaX = (int)(0.1 * xRed + 0.6 * xGreen + 0.1 * xBlue);
        // deltaY = (int)(0.1 * yRed + 0.6 * yGreen + 0.1 * yBlue); 
        deltaX[offset] = dX; 
        deltaY[offset] = dY; 
        result[offset] = (int)(sqrt((double)dX*dX +  
            (double)dY*dY) + 0.5);
    }
}

__global__ void suppression(int *pic, int *result, int* deltaX, int* deltaY, int xsize, int ysize){
    int j = blockIdx.x*blockDim.x + threadIdx.x; // col
    int i = blockIdx.y*blockDim.y + threadIdx.y; // row
    int offset = i*xsize + j; 

    //Skip edges 
    if(i > 0 && j > 0 && i < ysize-1 && j < xsize-1){
        float theta, mag1, mag2; 
        int DX = deltaX[offset]; 
        int DY = deltaY[offset]; 

        //No edge case 
        if(pic[offset] != 0){
            //DX >= 0 && DY >= 0 
            //45 degrees range
            if(DX >= 0 && DY >= 0){
                //DX - DY >= 0 
                if(DX - DY >= 0){
                    theta = (float)DY / DX;
                    mag1 = (1-theta)*pic[offset+1] + theta*pic[offset+xsize+1];
                    mag2 = (1-theta)*pic[offset-1] + theta*pic[offset-xsize-1];
                }
                //DX - DY < 0 
                else {
                    theta = (float)DX / DY;
                    mag1 = (1-theta)*pic[offset+xsize] + theta*pic[offset+xsize+1];
                    mag2 = (1-theta)*pic[offset-xsize] + theta*pic[offset-xsize-1];
                }
            }
            //DX >= 0 && DY < 0 
            //135 degrees 
            else if (DX >= 0 && DY < 0){
                // DX + DY >= 0 
                if(DX + DY >= 0){
                    theta = (float)-DY / DX;
                    mag1 = (1-theta)*pic[offset+1] + theta*pic[offset-xsize+1];
                    mag2 = (1-theta)*pic[offset-1] + theta*pic[offset+xsize-1];
                }
                //DX + DY < 0 
                else {
                    theta = (float)DX/ -DY;
                    mag1 = (1-theta)*pic[offset+xsize] + theta*pic[offset+xsize-1];
                    mag2 = (1-theta)*pic[offset-xsize] + theta*pic[offset-xsize+1];
                }
            }
            else if(DX < 0 && DY >= 0){
            //DX < 0 && DY >= 0 
            //90 degrees 
                //DX + DY >= 0 
                if(DX + DY >= 0){
                    theta = (float)-DX / DY;
                    mag1 = (1-theta)*pic[offset+xsize] + theta*pic[offset+xsize-1];
                    mag2 = (1-theta)*pic[offset-xsize] + theta*pic[offset-xsize+1];
                }
                //DX + DY < 0 
                else {
                    theta = (float)DY / -DX;
                    mag1 = (1-theta)*pic[offset-1] + theta*pic[offset+xsize-1];
                    mag2 = (1-theta)*pic[offset+1] + theta*pic[offset-xsize+1];
                }
            } 
            //DX < 0 && DY < 0
            //0 degrees 
            else {
                //DY - DX >= 0 
                if(DY - DX >= 0){
                    theta = (float)DY / DX;
                    mag1 = (1-theta)*pic[offset-1] + theta*pic[offset-xsize-1];
                    mag2 = (1-theta)*pic[offset+1] + theta*pic[offset+xsize+1];
                }
                //DY - DX < 0 
                else { 
                    theta = (float)DX / DY;
                    mag1 = (1-theta)*pic[offset-xsize] + theta*pic[offset-xsize-1];
                    mag2 = (1-theta)*pic[offset+xsize] + theta*pic[offset+xsize+1];
                }
            }
            //Mag checks 
            int pixelVal = pic[offset]; 
            if(pixelVal < mag1 && pixelVal < mag2) {
                result[offset] = 0;
            }
            else {
                result[offset] = pixelVal; 
            }
        }
        else {
            result[offset] = 0; 
        }
    }    
}


__device__ void line_trace(int *pic, bool *result, int offset, int lowT, int xsize) {
    unsigned n, s, e, w;
    unsigned nw, ne, sw, se;

    /* get indices */
    n = offset - xsize;
    nw = n - 1;
    ne = n + 1;
    s = offset + xsize;
    sw = s - 1;
    se = s + 1;
    w = offset - 1;
    e = offset + 1;

    if (pic[nw] >= lowT) {
        result[nw] = EDGE;
    }
    if (pic[n] >= lowT) {
        result[n] = EDGE;
    }
    if (pic[ne] >= lowT) {
        result[ne] = EDGE;
    }
    if (pic[w] >= lowT) {
        result[w] = EDGE;
    }
    if (pic[e] >= lowT) {
        result[e] = EDGE;
    }
    if (pic[sw] >= lowT) {
        result[sw] = EDGE;
    }
    if (pic[s] >= lowT) {
        result[s] = EDGE;
    }
    if (pic[se] >= lowT) {
        result[se] = EDGE;
    }
}

__global__ void hysteresis_high(int *pic, bool *result, int *strong_edges,  int highT, int xsize, int ysize)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x; // col
    int i = blockIdx.y*blockDim.y + threadIdx.y; // row
    int offset = i*xsize + j; 

    if(i >= 0 && j >= 0 && i < ysize && j < xsize){    
        if(pic[offset] > highT){
            strong_edges[offset] = 1; 
            result[offset] = EDGE; 
        }
        else{
            strong_edges[offset] = 0; 
            result[offset] = 0; 
        }
    }
}

__global__ void hysteresis_low(int *pic, bool *result, int *strong_edges, int lowT, int xsize, int ysize) {
    int j = blockIdx.x*blockDim.x + threadIdx.x; // col
    int i = blockIdx.y*blockDim.y + threadIdx.y; // row
    int offset = i*xsize + j; 

    //Skip edges 
    if(i > 0 && j > 0 && i < ysize-1 && j < xsize-1){
        if (1 == strong_edges[offset]){
            line_trace(pic, result, offset, lowT, xsize); 
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

main( int argc, char **argv ){
//   int thresh = DEFAULT_THRESHOLD;
  char *filename;
    filename = "img/img_4.ppm";
  
    if (argc > 1) {
        if (argc == 3)  { // filename AND threshold
            filename = strdup( argv[1]);
            // thresh = atoi( argv[2] );
        }
        if (argc == 2) { // default file but specified threshhold
        
            // thresh = atoi( argv[1] );
        }

        // fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
    }


    int xsize, ysize, maxval;
    Pixels * pic = read_ppm( filename, xsize, ysize, maxval ); 

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(ceil(xsize/ (float)BLOCK_SIZE_X ), ceil(ysize/ (float)BLOCK_SIZE_Y ));    

    int pixelNumBytes =  xsize * ysize * sizeof( Pixels ); 
    int intNumBytes = xsize * ysize * sizeof(int); 
    int mask_size = 7; 
    double *gauss_mask = gaussian_mask(); 

    //Timing
    cudaEvent_t start_event, stop_event;  
    float elapsed_time_par;

    //========Sequential Canny==================
    Pixels *stage1 = (Pixels *)malloc(pixelNumBytes); 
    int *delX = (int *)malloc(intNumBytes); 
    int *delY = (int *)malloc(intNumBytes); 
    int *stage2 = (int *)malloc(intNumBytes); 
    int *stage3 = (int *)malloc(intNumBytes); 
    int* sEdge = (int *)malloc(intNumBytes); 
    bool* stage4 = (bool *)malloc(xsize*ysize*sizeof(bool)); 

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    //==Stage 1==
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
    //==Stage 2=== 

    for(int i = 0; i < ysize; i++){
        for(int j = 0; j < xsize; j++){
            int offset = i*xsize + j; 
            if(i > 0 &&  i < (ysize-1)){
                int16_t xRed = 0;
                int16_t yRed = 0;
                int16_t xGreen = 0;
                int16_t yGreen = 0;
                int16_t xBlue = 0;
                int16_t yBlue = 0;
                int dX = 0; 
                int dY = 0; 
        
                //3 cases 
                //1st col 
                if((j % xsize) == 0){
                    xRed = (int16_t)(stage1[offset +1].r       - stage1[offset].r );
                    yRed = (int16_t)(stage1[offset + xsize].r  - stage1[offset].r ); 
        
                    xGreen = (int16_t)(stage1[offset +1].g      - stage1[offset].g );
                    yGreen = (int16_t)(stage1[offset + xsize].g - stage1[offset].g ); 
                    
                    xBlue = (int16_t)(stage1[offset +1].b      - stage1[offset].b );
                    yBlue = (int16_t)(stage1[offset + xsize].b - stage1[offset].b ); 
                }
                //last col 
                else if((j % xsize) == xsize -1){
                    xRed = (int16_t)(stage1[offset].r - stage1[offset - 1].r ); 
                    yRed = (int16_t)(stage1[offset].r - stage1[offset - xsize].r ); 
        
                    xGreen = (int16_t)(stage1[offset].g - stage1[offset - 1].g ); 
                    yGreen = (int16_t)(stage1[offset].g - stage1[offset - xsize].g ); 
        
                    xBlue = (int16_t)(stage1[offset].b  - stage1[offset - 1].b ); 
                    yBlue = (int16_t)(stage1[offset].b  - stage1[offset - xsize].b ); 
                }
                //middle col 
                else {
                    xRed = (int16_t)(stage1[offset + 1].r - stage1[offset - 1].r ); 
                    yRed = (int16_t)(stage1[offset + xsize].r - stage1[offset - xsize].r ); 
        
                    xGreen = (int16_t)(stage1[offset + 1].g  - stage1[offset - 1].g ); 
                    yGreen = (int16_t)(stage1[offset + xsize].g  - stage1[offset - xsize].g ); 
        
                    xBlue = (int16_t)(stage1[offset + 1].b  - stage1[offset - 1].b ); 
                    yBlue = (int16_t)(stage1[offset + xsize].b  - stage1[offset - xsize].b ); 
                }
        
                dX= (int)(0.2989 * xRed + 0.5870 * xGreen + 0.1140 * xBlue);
                dY= (int)(0.2989 * yRed + 0.5870 * yGreen + 0.1140 * yBlue); 
        
                // deltaX = (int)(0.1 * xRed + 0.6 * xGreen + 0.1 * xBlue);
                // deltaY = (int)(0.1 * yRed + 0.6 * yGreen + 0.1 * yBlue); 
                delX[offset] = dX; 
                delY[offset] = dY; 
                stage2[offset] = (int)(sqrt((double)dX*dX +  
                    (double)dY*dY) + 0.5);
            }
        }
    }


    //====Stage 3==========

    for(int i = 0; i < ysize; i++){
        for(int j = 0; j < xsize; j++){
            int offset = i*xsize + j; 

            if(i > 0 && j > 0 && i < ysize-1 && j < xsize-1){
                float theta, mag1, mag2; 
                int DX = delX[offset]; 
                int DY = delY[offset]; 
        
                //No edge case 
                if(stage2[offset] != 0){
                    //DX >= 0 && DY >= 0 
                    //45 degrees range
                    if(DX >= 0 && DY >= 0){
                        //DX - DY >= 0 
                        if(DX - DY >= 0){
                            theta = (float)DY / DX;
                            mag1 = (1-theta)*stage2[offset+1] + theta*stage2[offset+xsize+1];
                            mag2 = (1-theta)*stage2[offset-1] + theta*stage2[offset-xsize-1];
                        }
                        //DX - DY < 0 
                        else {
                            theta = (float)DX / DY;
                            mag1 = (1-theta)*stage2[offset+xsize] + theta*stage2[offset+xsize+1];
                            mag2 = (1-theta)*stage2[offset-xsize] + theta*stage2[offset-xsize-1];
                        }
                    }
                    //DX >= 0 && DY < 0 
                    //135 degrees 
                    else if (DX >= 0 && DY < 0){
                        // DX + DY >= 0 
                        if(DX + DY >= 0){
                            theta = (float)-DY / DX;
                            mag1 = (1-theta)*stage2[offset+1] + theta*stage2[offset-xsize+1];
                            mag2 = (1-theta)*stage2[offset-1] + theta*stage2[offset+xsize-1];
                        }
                        //DX + DY < 0 
                        else {
                            theta = (float)DX/ -DY;
                            mag1 = (1-theta)*stage2[offset+xsize] + theta*stage2[offset+xsize-1];
                            mag2 = (1-theta)*stage2[offset-xsize] + theta*stage2[offset-xsize+1];
                        }
                    }
                    else if(DX < 0 && DY >= 0){
                    //DX < 0 && DY >= 0 
                    //90 degrees 
                        //DX + DY >= 0 
                        if(DX + DY >= 0){
                            theta = (float)-DX / DY;
                            mag1 = (1-theta)*stage2[offset+xsize] + theta*stage2[offset+xsize-1];
                            mag2 = (1-theta)*stage2[offset-xsize] + theta*stage2[offset-xsize+1];
                        }
                        //DX + DY < 0 
                        else {
                            theta = (float)DY / -DX;
                            mag1 = (1-theta)*stage2[offset-1] + theta*stage2[offset+xsize-1];
                            mag2 = (1-theta)*stage2[offset+1] + theta*stage2[offset-xsize+1];
                        }
                    } 
                    //DX < 0 && DY < 0
                    //0 degrees 
                    else {
                        //DY - DX >= 0 
                        if(DY - DX >= 0){
                            theta = (float)DY / DX;
                            mag1 = (1-theta)*stage2[offset-1] + theta*stage2[offset-xsize-1];
                            mag2 = (1-theta)*stage2[offset+1] + theta*stage2[offset+xsize+1];
                        }
                        //DY - DX < 0 
                        else { 
                            theta = (float)DX / DY;
                            mag1 = (1-theta)*stage2[offset-xsize] + theta*stage2[offset-xsize-1];
                            mag2 = (1-theta)*stage2[offset+xsize] + theta*stage2[offset+xsize+1];
                        }
                    }
                    //Mag checks 
                    int pixelVal = stage2[offset]; 
                    if(pixelVal < mag1 && pixelVal < mag2) {
                        stage3[offset] = 0;
                    }
                    else {
                        stage3[offset] = pixelVal; 
                    }
                }
                else {
                    stage3[offset] = 0; 
                }
            }    
        }
    }

    //=====Stage 4=========

    int hT = 25; 
    int lT = 1; 

    for(int i = 0; i < ysize; i++){
        for(int j = 0; j < xsize; j++){
            int offset = i*xsize + j; 
            if(i >= 0 && j >= 0 && i < ysize && j < xsize){    
                if(stage3[offset] > hT){
                    sEdge[offset] = 1; 
                    stage4[offset] = EDGE; 
                }
                else{
                    sEdge[offset] = 0; 
                    stage4[offset] = 0; 
                }
            }        
        }
    }

    //===Stage 5======
    for(int i = 0; i < ysize; i++){
        for(int j = 0; j < xsize; j++){
            int offset = i*xsize + j; 
            if(i > 0 && j > 0 && i < ysize-1 && j < xsize-1){
                if (1 == sEdge[offset]){
                    unsigned n, s, e, w;
                    unsigned nw, ne, sw, se;
                
                    /* get indices */
                    n = offset - xsize;
                    nw = n - 1;
                    ne = n + 1;
                    s = offset + xsize;
                    sw = s - 1;
                    se = s + 1;
                    w = offset - 1;
                    e = offset + 1;
                
                    if (stage3[nw] >= lT) {
                        stage4[nw] = EDGE;
                    }
                    if (stage3[n] >= lT) {
                        stage4[n] = EDGE;
                    }
                    if (stage3[ne] >= lT) {
                        stage4[ne] = EDGE;
                    }
                    if (stage3[w] >= lT) {
                        stage4[w] = EDGE;
                    }
                    if (stage3[e] >= lT) {
                        stage4[e] = EDGE;
                    }
                    if (stage3[sw] >= lT) {
                        stage4[sw] = EDGE;
                    }
                    if (stage3[s] >= lT) {
                        stage4[s] = EDGE;
                    }
                    if (stage3[se] >= lT) {
                        stage4[se] = EDGE;
                    }
                
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
    free(stage3); 
    free(stage4); //stage 4 is the bool result 
    free(delX); 
    free(delY); 
    //========End Sequential Canny===============




    //CUDA data structures 
    int *deltaX             = NULL; 
    int *deltaY             = NULL; 
    Pixels *d_pic            = NULL; 
    Pixels *d_pixel_result   = NULL; 
    double *d_blur          = NULL; 
    int *d_int_result_0     = NULL;
    int *d_int_result_1     = NULL; 
    bool *d_bool_result     = NULL; 

    //CUDA MALLOC
    cudaMalloc((void**) &deltaX, intNumBytes);
    checkCudaError("malloc deltaX");

    cudaMalloc((void**) &deltaY, intNumBytes);
    checkCudaError("malloc deltaY");

    cudaMalloc((void**) &d_blur, sizeof(d_blur[0])*mask_size*mask_size);
    checkCudaError("malloc d_blur");

    cudaMalloc((void **) &d_pic, pixelNumBytes);
    checkCudaError("malloc d_pic");

    cudaMalloc((void **) &d_pixel_result, pixelNumBytes);
    checkCudaError("malloc d_pixel_result");

    cudaMalloc((void **) &d_int_result_0, intNumBytes);
    checkCudaError("malloc d_int_result_0");

    cudaMalloc((void **) &d_int_result_1, intNumBytes);
    checkCudaError("malloc d_int_result_1");

    cudaMalloc((void **) &d_bool_result, xsize*ysize*sizeof(bool));
    checkCudaError("malloc d_bool_result");


    //CUA MEMCPY 
    cudaMemcpy(d_blur, gauss_mask, sizeof(d_blur[0])*mask_size*mask_size, cudaMemcpyHostToDevice);
    checkCudaError("copy gauss_mask");

    cudaMemcpy(d_pic, pic, pixelNumBytes, cudaMemcpyHostToDevice);
    checkCudaError("copy pic");

    //Start timer 
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    // Guassian Filter: Pixel *d_pic, Pixel *d_pixel_result, int xsize, int ysize, double * d_blur, int mask_size 
    gauss_filter<<<grid, block>>>(d_pic, d_pixel_result, xsize, ysize, d_blur, mask_size); 
    checkCudaError("run gauss_filter");

    //Gradient: Pixel *d_pixel_result, int* d_int_result_0, int *deltaX, int *deltaY, int xsize, int ysize
    gradient_calc<<<grid, block>>>(d_pixel_result, d_int_result_0, deltaX, deltaY, xsize, ysize); 
    checkCudaError("run gradient_calc");

    //Suppresion: int* d_int_result_0, d_int_result_1, int* deltaX, int* deltaY, int xsize, int ysize 
    suppression<<<grid, block>>>(d_int_result_0, d_int_result_1, deltaX, deltaY, xsize, ysize); 
    checkCudaError("run suppression");

    //Threshold High: int* d_int_result_1, bool* d_bool_result, int* d_strong_edges, int highT, int xsize, int ysize
    // From Image 4 Run: Max Vale: 62; Min Vale: 1 
    int highT = 25; 
    int lowT = 1; 
    hysteresis_high<<<grid, block>>>(d_int_result_1, d_bool_result, d_int_result_0, highT, xsize, ysize); 
    checkCudaError("run hysteresis_high");

    //Threshold Low: int* d_int_result_1, bool* d_bool_result, int *strong_edges, int lowT, int xsize, int ysize 
    hysteresis_low<<<grid, block>>>(d_int_result_1, d_bool_result, d_int_result_0, lowT, xsize, ysize); 
    checkCudaError("run hysteresis_low");

    //END TIMER 
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Edge Detection Parallel Runtime: %f ms\n", elapsed_time_par);

    dim3 grid2(ceil((xsize/16)/ (float)BLOCK_SIZE_X ), ceil((ysize/16)/ (float)BLOCK_SIZE_Y )); 

    //gcount 
    int *num_pix_found;
    cudaMallocManaged(&num_pix_found, 4); // allocate space for num_pix_found on device
    *num_pix_found = 0;

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);


    gcount<<<grid2, block>>>(d_pic, d_bool_result, d_pixel_result, xsize, ysize, num_pix_found); 
    checkCudaError("run gcount");

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_par, start_event, stop_event);

    fprintf(stderr, "   Pixel Count Parallel Runtime: %f ms\n", elapsed_time_par);

    fprintf(stderr, "   file: %s, num_pix_found: %d, cm^2: %d\n",filename, *num_pix_found, *num_pix_found / 467); // there are 466.667 pixels per cm^2

    Pixels * result = (Pixels *)malloc(pixelNumBytes); 
    cudaMemcpy(result, d_pixel_result, pixelNumBytes, cudaMemcpyDeviceToHost);
    checkCudaError("copy result");

    write_ppm( "canny.ppm", xsize, ysize, 255, result);

    // fprintf(stderr, "Finished!!!!!!\n");
    
}
