package lv.edi.EDI_3DAtA.bloodvesselsegm;

import org.ejml.data.DenseMatrix64F;

/**
 * 
 * @author Ričards Cacurs
 *
 * Class consisting of operations for filtering image in DenseMatrix64F format
 */
public class FilteringOperations {
	
	/**
	 * Functions performs 2D convolution on image given in DenseMatrix64F format
	 * @param inputImage image on which to perform convolution
	 * @param kernel filter kernel. Filter is considered to be centered in floor((kernelHeight-1)/2)
	 * 																	   floor((kernelWidth-1)/2) 
	 * @return DenseMatrix64F type image same size as input image
	 */
	public static DenseMatrix64F convolve2D(DenseMatrix64F inputImage, DenseMatrix64F kernel){
		
		DenseMatrix64F output = new DenseMatrix64F(inputImage.numRows, inputImage.numCols);
		int kernelCenterR = (int)Math.floor((kernel.numRows-1)/2);
		int kernelCenterC = (int)Math.floor((kernel.numCols-1)/2);
		double sample;
		double samplesum;
		for(int i=0; i<inputImage.numRows; i++){
			for(int j=0; j<inputImage.numCols; j++){
				
				samplesum=0;
				for(int ki=0; ki<kernel.numRows; ki++){
					for(int kj=0; kj<kernel.numCols; kj++){
						int pixelRowIndex = i+(ki-kernelCenterR);
						int pixelColIndex = j+(kj-kernelCenterC);
						if((pixelRowIndex<0)|| //checking if not out of bounds
						   (pixelRowIndex>=inputImage.numRows)||
						   (pixelColIndex<0)||
						   (pixelColIndex>=inputImage.numCols)){
							sample=0;
						} else{
							sample = inputImage.get(pixelRowIndex, pixelColIndex);
						}
						samplesum+=sample*kernel.get(kernel.numRows-1-ki, kernel.numCols-1-kj); //kernel is flipped
					}
				}
				output.set(i, j, samplesum);
			}
		}
		return output;
	}

}
