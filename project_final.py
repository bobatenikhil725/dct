# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:20:56 2021

@author: bobate
"""
#importing required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
#Definig fuctions 

#Zigzag scanning of matrix
def zigzagscan(input):
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]  
    #print(vmax ,hmax )
    i = 0
    output = np.zeros(( vmax * hmax))
    #----------------------------------
    while ((v < vmax) and (h < hmax)):    	
        if ((h + v) % 2) == 0:                 # going up           
            if (v == vmin):
            	#print(1)
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
            	#print(2)
            	output[i] = input[v, h] 
            	v = v + 1
            	i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
            	#print(3)
            	output[i] = input[v, h] 
            	v = v - 1
            	h = h + 1
            	i = i + 1      
        else:                                    # going down

        	if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
        		#print(4)
        		output[i] = input[v, h] 
        		h = h + 1
        		i = i + 1
        
        	elif (h == hmin):                  # if we got to the first column
        		#print(5)
        		output[i] = input[v, h] 

        		if (v == vmax -1):
        			h = h + 1
        		else:
        			v = v + 1

        		i = i + 1

        	elif ((v < vmax -1) and (h > hmin)):     # all other cases
        		#print(6)
        		output[i] = input[v, h] 
        		v = v + 1
        		h = h - 1
        		i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
        	#print(7)        	
        	output[i] = input[v, h] 
        	break
    return output
#ZIgzag seq array to matrix
def reverse_zigzag(input, vmax, hmax):
	
	#print input.shape

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0
    #----------------------------------

	while ((v < vmax) and (h < hmax)): 
		#print ('v:',v,', h:',h,', i:',i)   	
		if ((h + v) % 2) == 0:                 # going up
            
			if (v == vmin):
				#print(1)
				
				output[v, h] = input[i]        # if we got to the first line

				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1

        
		else:                                    # going down

			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
        
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
        		        		
			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1




		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[v, h] = input[i] 
			break


	return output

# Creating tree nodes
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)

#Huffman code tree
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is np.float64 or type(node) is int :
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d

#Reading image and converting to greyscale
img = cv2.imread('C:/Users/bobat/lenna_grey.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img=np.float32(gray_img)
#quantization
Q=10

#padding by zero to get perfect blocks of 8x8
r,c=0,0
row,col=gray_img.shape
if row%8!=0:
    r=8-row%8
else :    r=0
if col%8!=0:
    c=8-col%8
else :
    c=0
gray_img=np.pad(gray_img, ((0,r),(0,c)), 'constant')

#Dividing into block and applying dct          
T1=time.time()
row,col=gray_img.shape
for i in range (0,int(row/8)):
    for j in range (0,int(col/8)):
        locals()["BR"+str(i)+"BC"+str(j)]=gray_img[(i*8):(i*8+8),(j*8):((j*8)+8)]
        #Applying dct
        locals()["DCTR"+str(i)+"DCTC"+str(j)]=((cv2.dct(locals()["BR"+str(i)+"BC"+str(j)]))/Q).astype(int)

#flattening DCT coefficents using zig-zag scan
seq=[]
for i in range (0,int(row/8)):
    for j in range (0,int(col/8)):
        seq=np.append(seq,zigzagscan((locals()["DCTR"+str(i)+"DCTC"+str(j)])))

#runlengthcode of flatend sequence
rle=[]
count=0
i=0
flag=True
while (i<seq.shape[0]):
    
    if flag==True:
        if seq[i]!=0:
            rle.append(seq[i])
            count=0
        else:
            count=count+1
            flag=False
    else:
        if seq[i]!=0:
            rle.append(0)
            rle.append(count)
            rle.append(seq[i])
            count=0
            flag=True   
        else:
            count=count+1
            flag=False
    i=i+1

#Getting symbols and their freqency and further arranging in descending order
counter=collections.Counter(rle)
symbol_freq=sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
nodes = symbol_freq

#Generating huffman tree
while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

#Getting huffman code for each symbol
huffmanCode = huffman_code_tree(nodes[0][0])

#Generating Bitstring of the sequence

BitString=""
for i in range(len(rle)):
    BitString=BitString+"{}".format(huffmanCode[rle[i]])
encoded_bit_length=len(BitString)
wo_enc_bit_length=row*col*8
bits_symbol=encoded_bit_length/(row*col)

#exporting bitstring as txt file
text_file = open("Output_bitstring.txt", "w")
text_file.write(BitString)
text_file.close()
T2=time.time()


#DECODING 
#decoding rle from bitstring
decoded_rle=[]
buffer=""
for i in range (encoded_bit_length):
    buffer=buffer+BitString[i]
    for (sym,code) in huffmanCode.items():
        if buffer==code:
            decoded_rle=np.append(decoded_rle,sym)
            buffer=""

#Decoding sequence from rle
decoded_seq=[]
i=0
while i < (len(decoded_rle)):
    if decoded_rle[i]==0:
        temp=[0]*(decoded_rle[i+1]).astype(int)
        decoded_seq=np.append(decoded_seq,temp)
        i=i+2
    else:
        decoded_seq=np.append(decoded_seq,decoded_rle[i])
        i=i+1

pad=len(seq)-len(decoded_seq)
decoded_seq=np.append(decoded_seq,[0]*pad)

#GETTING 8X8 dct BLOCK FROM SEQUENCE
for i in range (0,int(row*col/(8*8))):
     locals()["MAT"+str(i)]=reverse_zigzag(decoded_seq[i*64:i*64+64], 8, 8)

#applying Inverse DCT and dequantization
for i in range (0,int(row*col/(8*8))):
     locals()["IMAT"+str(i)]=(cv2.idct(np.float32(((locals()["MAT"+str(i)]))*Q))).astype(np.uint8)

#Reconstructing image
for i in range (0,int(row/8)):
    locals()["ROW"+str(i)]=locals()["IMAT"+str(i*64)]
    for j in range (1,int(col/8)):
        locals()["ROW"+str(i)]=np.c_[locals()["ROW"+str(i)],locals()["IMAT"+str(i*64+j)]]

reconstructed_img=ROW0
for i in range (1,int(row/8)):
    reconstructed_img=np.r_[reconstructed_img,locals()["ROW"+str(i)]]
T3=time.time()

Mean_sq_Error=np.sum((gray_img-reconstructed_img)**2)/(int(row/8)*int(col/8))

plt.imshow(reconstructed_img,cmap='gray')
plt.title('RECONSTRUCTED IMAGE')
plt.show()

plt.imshow(gray_img,cmap='gray')
plt.title('INPUT IMAGE')
plt.show()

print("Total number of bits in input image:{}".format(wo_enc_bit_length))
print("Total number of bits after encoding:{}".format(encoded_bit_length))
print("Compression Ratio:({})/({})={}".format(wo_enc_bit_length,encoded_bit_length,wo_enc_bit_length/encoded_bit_length))
print("RMSE : {}".format(np.sqrt(Mean_sq_Error)))
print("PSNR : {}".format(cv2.PSNR((gray_img).astype(np.uint8), reconstructed_img)))
print("Time required for compression:{}".format(T2-T1))
print("Time required for decompression:{}".format(T3-T2))
