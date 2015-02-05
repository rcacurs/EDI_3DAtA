package lv.edi.EDI_3DAtA.bloodvesselsegm;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
SMFilterBlock - filter block of the Stacked Multiscale Feature Extractor
*/
public class SMFilterBlock {
	
	/**
	 * Function returns the image which is filtered with filter bank
	 * @param inputImage - input 2D image in the DenseMatrix64F format
	 * @param patchSize - size of the patch
	 * @param dCodesFileName - file name of the filter bank
	 * @param dMeanFileName - file name of the filter bank mean values
	 * @return - filtering result
	 */
	public static DenseMatrix64F filter ( DenseMatrix64F inputImage, int patchSize, DenseMatrix64F codes, DenseMatrix64F mean )
	{
		
		inputImage = SMFilterBlock.padArray( inputImage, (int) patchSize/2 ); // pad original image with zeros
		
		DenseMatrix64F allPatches = SMFilterBlock.im2col( inputImage, patchSize ); // get all patches
		
		allPatches = SMFilterBlock.bsxfunMinus( allPatches, SMFilterBlock.computeMeanOfPatches( inputImage, patchSize ) ); // contract normalization
		
		DenseMatrix64F varVec = SMFilterBlock.arrayRowVar(allPatches);
		
		varVec = SMFilterBlock.smNormalizer(varVec);
		
		allPatches = SMFilterBlock.bsxfunDivide(allPatches, varVec);
		
		varVec = null;
		
		allPatches = SMFilterBlock.bsxfunMinus(allPatches, mean);

		
		DenseMatrix64F multResult = new DenseMatrix64F( allPatches.numRows, codes.numCols );

		CommonOps.mult( allPatches, codes, multResult );
		
		return multResult;
		
	}
	
	/**
	 * computes the normalising vector for patches normalisation 
	 * @param vector - vector of variances to be updated
	*/
	public static DenseMatrix64F smNormalizer ( DenseMatrix64F vector )
	{
		
		for ( int count = 0; count < vector.numRows; count++ )
		{
			vector.data[count] = Math.sqrt( vector.data[count] + 10 );
		}
		return vector;
	}
	
	/**
	 * computes the variance of the row elements in the input mean normalised array. The normaliser is (array.numCols-1)
	 * @param array - array to be subtracted from
	 * @param vector - vector to subtract from the array
	*/
	public static DenseMatrix64F arrayRowVar ( DenseMatrix64F array )
	{
		DenseMatrix64F varVec = new DenseMatrix64F(array.numRows, 1);
		
		for ( int row = 0; row < array.numRows; row++ )
		{
			for ( int col = 0; col < array.numCols; col++ )
			{
				varVec.data[row] += array.get(row, col) * array.get(row, col);
			}
			varVec.data[row] = varVec.data[row]/(array.numCols-1);
		}
		return varVec;
	}
	
	
	/**
	 * element-wise division of input array rows/columns with elements in the input vector
	 * @param array - array to be divided and updated
	 * @param vector - vector with denominator values
	*/
	public static DenseMatrix64F bsxfunDivide ( DenseMatrix64F array, DenseMatrix64F vector )
	{	
		if ( array.getNumRows() == vector.getNumElements() )
		{			
			for ( int col = 0; col < array.numCols; col++ )
			{
				for ( int row = 0; row < array.numRows; row++ )
				{
					array.set(row, col, array.get(row, col) / vector.data[row] );
				}
			}
		}
		else
		{
			
		}
		return array;
		
	}
	
	
	/**
	 * Subtracts a vector from the matrix row-wise or column-wise depending on the size of the vector
	 * @param array - array to be subtracted from
	 * @param vector - vector to subtract from the array
	 * @return - 
	*/
	public static DenseMatrix64F bsxfunMinus ( DenseMatrix64F array, DenseMatrix64F vector )
	{
		
		if ( array.getNumRows() == vector.getNumElements() )
		{			
			for ( int col = 0; col < array.numCols; col++ )
			{
				for ( int row = 0; row < array.numRows; row++ )
				{
					array.add( row, col, - vector.data[row] );
				}
			}
		}
		else
		{
			for ( int col = 0; col < array.numCols; col++ )
			{
				for ( int row = 0; row < array.numRows; row++ )
				{
					array.add( row, col, - vector.data[col] );
				}
			}
		}
		return array;
		
	}
	
	/**
	 * Rearrange image blocks into rows. Similar to Matlab im2col
	 * @param inputImage - input image in the DenseMatrix64F format
	 * @param patchSize - defines the size of the patch. Patch dimensions: (patchSize x patchSize)
	*/
	public static DenseMatrix64F im2col ( DenseMatrix64F inputImage, int patchSize )
	{
		int pixNum = patchSize*patchSize; // number of pixels in the patch
		int count = 0;
		int colMax = inputImage.getNumCols() - 2*(patchSize/2); // max number of patches in X direction
		int rowMax = inputImage.getNumRows() - 2*(patchSize/2); // max number of patches in Y direction
		int patchesNum = colMax * rowMax ; // total number of patches in the image
		
		DenseMatrix64F patch = new DenseMatrix64F(patchSize, patchSize); // array to store the patch
		DenseMatrix64F allPatches = new DenseMatrix64F(patchesNum, patchSize*patchSize); // array to store all patches in vectorized form
		
		for ( int col = 0; col < colMax; col++ )
		{
			for ( int row = 0; row < rowMax; row++ )
			{
				patch = CommonOps.extract( inputImage, row, row + patchSize, col, col + patchSize ); // get current patch
				CommonOps.transpose(patch); // transpose the patch
				patch.reshape( 1 , pixNum , false ); // convert it to the vector form
				
				CommonOps.insert(patch, allPatches, count, 0); // save the patch to the array with all the patches
				count++;
				
			}
		}
		return allPatches;
		
	}
		
	/**
	 * Computes the vector with mean values of all patches in the input image
	 * @param inputImage - input image in the DenseMatrix64F format
	 * @param patchSize - defines the size of the patch. Patch dimensions: (patchSize x patchSize)
	*/
	public static DenseMatrix64F computeMeanOfPatches ( DenseMatrix64F inputImage, int patchSize )
	{
		
		int pixNum = patchSize*patchSize; // number of pixels in the patch
		int count = 0;
		int colMax = inputImage.getNumCols() - 2*(patchSize/2); // max number of patches in X direction
		int rowMax = inputImage.getNumRows() - 2*(patchSize/2); // max number of patches in Y direction
		int patchesNum = colMax * rowMax; // total number of patches in the image
		
		DenseMatrix64F meanVector = new DenseMatrix64F( patchesNum, 1 ); // vector to store mean values
		
		DenseMatrix64F integralImage = SMFilterBlock.integralImage( inputImage ); // Compute the integral image
		
		for ( int col = 0; col < colMax; col++ )
		{
			for ( int row = 0; row < rowMax; row++ )
			{
				meanVector.set(count, 0, SMFilterBlock.regionSum( integralImage, row, col, patchSize, patchSize )/pixNum );
				count++;
			}
		}
		return meanVector;
	}
	
	/**
	 * Computes the integral image of the input 2D matrix
	 * @param inputImage - input 2D matrix of the DenseMatrix64F type
	*/
	public static DenseMatrix64F integralImage( DenseMatrix64F inputImage )
	{
		
		DenseMatrix64F integralImage = inputImage.copy();
		
		double prevrow = 0;
		double prevcol = 0;
		double prevRC = 0;
		
		int rowMax = inputImage.getNumRows();
		int colMax = inputImage.getNumCols();
		
		for ( int row = 0; row < rowMax; row++ )
		{
			for ( int col = 0; col < colMax; col++ )
			{
				prevrow = 0;
				prevcol = 0;
				prevRC = 0;
				
				if ( row > 0 && col > 0 ) { prevRC = integralImage.get( row-1, col-1 ); }
				
				if ( row > 0 ) { prevrow = integralImage.get( row-1, col ); }  // get(int row, int col)
				
				if ( col > 0 ) { prevcol = integralImage.get( row, col-1 ); }
				
				integralImage.set( row, col, prevrow + prevcol + inputImage.get( row, col ) - prevRC );
			}
		}
		
		return integralImage;
	}
	
	/**
	 * Computes the sum of the elements in the region specified by the parameters
	 * @param integralImage - integral image in the DenseMatrix64F format
	 * @param row - row number / Y coordinate of the top left corner
	 * @param col - col number / X coordinate of the top left corner
	 * @param h - height o the region
	 * @param w - width of the region
	*/
	public static double regionSum( DenseMatrix64F integralImage, int row, int col, int h, int w )
	{
		double sum = 0;
		row--; col--;
		if ( row >= 0 && col >= 0 )
		{
			sum = integralImage.get( row, col ) + integralImage.get( row + h, col + w ) 
				- integralImage.get( row + h, col ) - integralImage.get( row, col + w );
		}
		else if ( row == -1 && col >= 0 )
		{
			sum = integralImage.get( row + h, col + w ) 
					- integralImage.get( row + h, col );
		}
		else if ( col == -1 && row >= 0 )
		{
			sum = integralImage.get( row + h, col + w ) 
					- integralImage.get( row, col + w );
		}
		else
		{
			sum = integralImage.get( row + h, col + w );
		}
		
		return sum ;
	}
	
	/**
	 * Pads the array with zeros 
	 * @param inputImage - input 2D matrix of the DenseMatrix64F type
	 * @param padSize - padding size. The size of the output array = ( Height + 2*padSize ) x ( Width + 2*padSize )
	*/
	public static DenseMatrix64F padArray( DenseMatrix64F inputImage, int padSize )
	{
		// Create a new Matrix with the specified shape whose elements initially have the value of zero:
		DenseMatrix64F paddedArray = new DenseMatrix64F( inputImage.numRows + 2*padSize, inputImage.numCols + 2*padSize );
		
		// insert(ReshapeMatrix64F src, ReshapeMatrix64F dest, int destY0, int destX0):
		CommonOps.insert( inputImage, paddedArray, padSize, padSize );
		
		return paddedArray; 
	}
	
}
