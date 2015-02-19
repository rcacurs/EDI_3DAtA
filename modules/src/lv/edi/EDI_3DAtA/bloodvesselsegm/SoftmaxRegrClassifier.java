package lv.edi.EDI_3DAtA.bloodvesselsegm;

import lv.edi.EDI_3DAtA.common.DenseMatrixConversions;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * Softmax regression classifier class
 * @author Olegs
 */
public class SoftmaxRegrClassifier {
	
	private DenseMatrix64F mean; // vector for mean-normalization
	private DenseMatrix64F sd; // vector for sd-normalization
	private DenseMatrix64F data; // the data to be classified in the DenseMatrix64F format
	private DenseMatrix64F model; // classifier parameters
	private DenseMatrix64F result;
	private int imHeight = 0; // Height of the output image / array
	private int imWidth = 0; // Width of the output image / array
	private DenseMatrix64F maskImage; // Mask image
	
	/**
	 * Constructor of the LogRegrClassifier class
	 * @param meanFilename - name of the CSV file for mean-normalization of the features
	 * @param sdFilename - name of the CSV file for sd-normalization of the features
	 */
	public SoftmaxRegrClassifier(String meanFilename, 
			String sdFilename, 
			String modelFilename, 
			int imHeight, int imWidth)
	{
		this.mean = DenseMatrixConversions.loadCSVtoDenseMatrix( meanFilename );
		this.sd = DenseMatrixConversions.loadCSVtoDenseMatrix( sdFilename );
		this.model = DenseMatrixConversions.loadCSVtoDenseMatrix( modelFilename );
		this.imHeight = imHeight;
		this.imWidth = imWidth;
		// this.maskImage = maskImage;
	}
	
	/**
	 * set the mask for the current layer
	 * @param image - image of the mask
	 */
	public void setMaskImage(DenseMatrix64F image)
	{
		this.maskImage = image;
	}
	
	/**
	 * set the data to be classified 
	 * @param layerFeatures - array with feature vectors for each pixel in the layer/image
	 */
	public void setData (LayerSMFeatures layerFeatures)
	{
		this.data = new DenseMatrix64F( layerFeatures.getFeatures() );
		CommonOps.transpose(data);
	}
	
	/**
	 * The main classification method
	 */
	public void classify()
	{
		
		/* for ( int row = 0; row < data.numRows; row++ )
		{
			for ( int col = 0; col < data.numCols; col++ )
			{
				data.set(row, col, 0.00001*row + 0.01*col);
			}
		} */
		
		data = SMFilterBlock.bsxfunMultiply(data, maskImage); // mask the image
		
		// DenseMatrixConversions.saveDenseMatrixToCSV( data, "../../modules/res/data" );
		
		this.normalize(); // Normalise the data before classification 
		this.addLastColumnOfOnes(); // add column of ones
		
		DenseMatrix64F multResult = new DenseMatrix64F( new double[2][data.numRows] ); // matrix to store the multiplication result
		
		CommonOps.transpose(data); // transpose the data before multiplication
		
		CommonOps.mult(model, data, multResult); // multiply model with data
		
		//DenseMatrix64F maxVec = new DenseMatrix64F( 1, multResult.numCols );
		DenseMatrix64F maxVec = columnMax(multResult);
		
		multResult = SMFilterBlock.bsxfunMinus(multResult, maxVec); // subtract max column values column-wise
		maxVec = null; // this vector is not needed any more
		
		multResult = arrayExp( multResult );
		
		DenseMatrix64F sumVec = colunWiseSum( multResult );
		
		multResult = SMFilterBlock.bsxfunDivide(multResult, sumVec);
		sumVec = null;
		
		// DenseMatrix64F imVecFormat = columnMax(multResult);
		
		DenseMatrix64F imVecFormat = getSecondRow(multResult);
		
		setClassificationResult(imVecFormat);
		
		// CommonOps.transpose(multResult);
		// DenseMatrixConversions.saveDenseMatrixToCSV( new DenseMatrix64F(result), "../../modules/res/result" );
		
	}
	
	/**
	 * Set the elements in the 2D array
	 * @param vector to be saved to the 2D array
	 */

	private void setClassificationResult(DenseMatrix64F vec)
	{
		double[][] result = new double[imHeight][imWidth];
		
		int count = 0;
		for ( int row = 0; row < imHeight; row++ )
		{
			for ( int col = 0; col < imWidth; col++ )
			{
				result[row][col] = vec.get(count);
				count++;
			}
		}
		
		this.result = new DenseMatrix64F(result);
		
	}
	
	/**
	 * Returns sums of the elements in the columns on the input array
	 * @param array - input 2D array
	 * @return - vector of column-wise sums
	 */
	private DenseMatrix64F colunWiseSum ( DenseMatrix64F array )
	{
		
		DenseMatrix64F sumVec = new DenseMatrix64F( 1, array.numCols );
		
		for ( int col = 0; col < array.numCols; col++ )
		{
			for ( int row = 0; row < array.numRows; row++ )
			{
				sumVec.set(  col,  sumVec.get(col) + array.get(row, col)  );
			}
		}
		return sumVec;
	}
	
	/**
	 * Returns a second row of the (2 x N)-dimensional Array
	 * @param array - input 2D array of the (2 x N) size
	 * @return the second row
	 */
	private DenseMatrix64F getSecondRow (DenseMatrix64F array)
	{
		DenseMatrix64F secondRow = new DenseMatrix64F( 1, array.numCols );
		
		for ( int col = 0; col < array.numCols; col++ )
		{
			secondRow.set(0, col, array.get(1, col) );
		}
		
		return secondRow;
	}
	
	/**
	 * Return the exp() value for each element in the array
	 * @param array - input array
	 * @return - array of exp() for each element in the input array
	 */
	private DenseMatrix64F arrayExp (DenseMatrix64F array)
	{
		int iMax = array.getNumElements();
		for ( int i = 0; i < iMax; i++ )
		{
			array.set(  i, Math.exp( array.get(i) )  );
		}
		return array;
	}
	
	/**
	 * Returns a maximum element for each column in the input (2 x N) Array
	 * @param array - input 2D array of the (2 x N) size
	 * @return a vector of max values in the columns
	 */
	private DenseMatrix64F columnMax (DenseMatrix64F array)
	{
		DenseMatrix64F maxVec = new DenseMatrix64F( 1, array.numCols );
		
		for ( int col = 0; col < array.numCols; col++ )
		{
			if ( array.get(0, col) >= array.get(1, col) )
			{
				maxVec.set(0, col, array.get(0, col) );
			}
			else
			{
				maxVec.set(0, col, array.get(1, col) );
			}
		}
		
		return maxVec;
	}
	
	/**
	 * perform the mean and sd normalisation of the data to be classified
	 */
	private void normalize()
	{
		SMFilterBlock.bsxfunMinus(data, mean); // mean-normalization
		SMFilterBlock.bsxfunDivide(data, sd); // sd-normalization
	}
	
	/**
	 * add column of ones at the end of the array
	 */
	private void addLastColumnOfOnes()
	{	
		DenseMatrix64F temp = new DenseMatrix64F( new double[data.numRows][data.numCols+1] ); 
		CommonOps.insert(data, temp, 0, 0);
		
		int col = data.numCols;
		for ( int row = 0; row < data.numRows; row++ )
		{
			temp.set( row, col, 1 ); // set the entries in the last column to 1
		}
		this.data = temp;
	}
	
	/**
	 * This function returns the classification result
	 * @return  - classification result - 2D array
	 */
	public DenseMatrix64F getResult()
	{
		return this.result;
	}

}
