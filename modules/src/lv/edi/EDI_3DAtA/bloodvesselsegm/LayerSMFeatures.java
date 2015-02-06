package lv.edi.EDI_3DAtA.bloodvesselsegm;

import org.ejml.data.DenseMatrix64F;

/**
 * Feature vectors and parameters of a layer/image
 * @author Olegs
 */
public class LayerSMFeatures {
	
	private float[][] features;
	private int numRows = 0;
	private int numCols = 0; 
	private int numFeatures = 0;
	
	/**
	 * object constructor of the LayerFeatures class
	 * @param numRows - number of rows in the layer
	 * @param numCols - number of columns in the layer
	 * @param numFeatures - dimensionality of the feature vector
	 */
	public LayerSMFeatures(int numRows, int numCols, int numFeatures)
	{
		this.numRows = numRows;
		this.numCols = numCols;
		this.numFeatures = numFeatures;
		features = new float[numRows*numCols][numFeatures]; // Initialise an array to store the feature vectors
	}
	
	
	/**
	 * Fast method to set the element in the feature array. Type casting (double -> float) is done internally
	 * NOTE: the requesting program should scan the array in column-major order
	 * @param idx - index of the pixel in the column-major order
	 * @param featureNum - serial number of the feature
	 * @param value - feature value / value to set. Type: double.
	 */
	public void setFeatureFast( int idx, int featureNum, double value )
	{
		features[ idx ][ featureNum ] = (float) value;
	}
	
	/**
	 * Set the element in the feature array. Type casting (double -> float) is done internally
	 * @param row - current pixel row number
	 * @param col - current pixel column number
	 * @param featureNum - serial number of the feature
	 * @param value - feature value / value to set. Type: double.
	 */
	public void setFeature( int row, int col, int featureNum, double value )
	{
		int rowIndex = col * numRows + row; // vertical position of current feature vector
		features[ rowIndex ][ featureNum ] = (float) value;
	}
	
	/**
	 * returns a single feature with specified serial number for the current index
	 * NOTE: the requesting program should scan/fill the array in column-major order
	 * @param idx - index of the pixel in the column-major order
	 * @param featureNum - serial number of the feature
	 * @return a feature with specified serial number (float type)
	 */
	public float getFeatureFast( int idx, int featureNum )
	{	
		float feature = features[idx][featureNum]; // feature vectors are stored as rows
		return feature;
	}
	
	/**
	 * returns a single feature with specified serial number for the current location/pixel
	 * @param row - row number of the pixel
	 * @param col - column number of the pixel
	 * @param featureNum - serial number of the feature
	 * @return a feature with specified serial number (float type)
	 */
	public float getFeature( int row, int col, int featureNum )
	{
		int rowIndex = col * numRows + row; // vertical position of current feature vector
		
		float feature = features[rowIndex][featureNum]; // feature vectors are stored as rows 
		
		return feature;
	}
	
	/**
	 * returns a feature vector for the current location/pixel
	 * @param row - row number of the pixel
	 * @param col - column number of the pixel
	 * @return a feature vector of the float type
	 */
	public float[] getFeatureVector( int row, int col )
	{
		float[] vector = new float[numFeatures];
		
		int rowIndex = col * numRows + row; // vertical position of current feature vector
		
		for ( int j = 0; j < features[0].length; j++ )
		{
			vector[j] = features[rowIndex][j]; // feature vectors are stored as rows 
		}
		
		return vector;
		
	}
	
	/**
	 * Get the number of rows in the layer/image
	 * @return the number of rows
	 */
	public int getNumRows()
	{
		return this.numRows;
	}
	
	/**
	 * Get the number of columns in the layer/image
	 * @return the number of columns
	 */
	public int getNumCols()
	{
		return this.numCols;
	}

	/**
	 * Get the dimensionality of the feature vector
	 * @return the number of features in the feature vector
	 */
	public int getNumFetures()
	{
		return this.numFeatures;
	}
	
	/**
	 * Return feature vectors for the current layer
	 * @return array of feature vectors of the float[][] type
	 */
	public float[][] getFeatures()
	{
		return features;
	}
	
	/**
	 * Returns specific feachure image ad DenseMatrix64F
	 * @param featureIndex feature index to return 
	 * @return DenseMatrix64F specified feature in DenseMatrix64F format
	 */
	public DenseMatrix64F getFeature(int featureIndex){
		DenseMatrix64F feature = new DenseMatrix64F(numRows, numCols);
		int pixelIndex=0;
		for(int i=0; i<numCols; i++){
			for(int j=0; j<numRows; j++){
				feature.unsafe_set(j, i, features[pixelIndex][featureIndex]);
				pixelIndex++;
			}
		}
		return feature;
	}
	
}
