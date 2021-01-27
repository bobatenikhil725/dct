In Transform coding, image is transformed into different domain from spatial domain and then the transformed coefficients are encoded. Transform coding helps to get high data compression at the expense of higher computational requirement. There are various transforms but DCT is considered superior because (a)It has better energy compaction(b)It is faster(c)It is optimized (d)It is not signal dependent

# The discrete cosine transform 
(DCT) converts image in spatial domain to frequency domain. In DCT the rows of the N × N transform matrix C are obtained as a function of cosines. The amount of variation increases as we move down the rows.

# Methodology
ENCODER\
Step(1):Input image is divided into 8x8 blocks.\
Step(2):Apply DCT to each block & quantize it by scalar quantizer Q-10.\
Step(3):Zigzag Scan is performed\
Step(4):Run Length coding is done to further reduce array size.\
Step(5):With the help of Huffman coding, code is generated for each symbol.\
Step(6):By using code for symbols given input sequence is converted into bitstream and it is output of encoder.

DECODER\
Step(7):Decoding part starts from this step. Input bitstream is converted back to sequence of symbol with help of Huffman code table and we obtain RLE code as output.\
Step(8):RLE code sequence is further converted back original sequence.\
Step(9):First 64 elements of sequence are taken and 8x8 matrix is obtained by inverse zigzag scanning method. This matrix is quantized DCT coefficient of First block. Similar procedure is applied to get back all the blocks.\
Step(10):Further the matrix is dequantized by multiplying scalar (Q-10) and inverse DCT is applied to get first block of reconstructed image.\
Step(11):All the blocks are stacked together to get back reconstructed image.

# RESULTS 
Total number of bits in input image: 2097152bits\
Total number of bits after encoding: 320099bits\
Compression Ratio:(2097152)/(320099)=6.552\
RMSE : 31.545\
PSNR: 36.214db

# Changes to make
Change the path for given input image at line no. 193 in code.\
Quantization value can also be changed by varying Q at line no.

