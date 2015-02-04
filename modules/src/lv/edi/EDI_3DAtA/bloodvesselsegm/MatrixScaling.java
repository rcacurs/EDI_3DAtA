package lv.edi.EDI_3DAtA.bloodvesselsegm;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * 
 * @author Ričards Cacurs
 * 
 * Functions for matrix scaling and interpolation
 *
 */
public class MatrixScaling {
	
	/**
	 * Function creates larger larger matrix specified by scale factor.
	 * Currently supports integer scale factor
	 * @param matrix DenseMatrix64F input matrix
	 * @param scale integer scaling factor
	 * @return DenseMatrix64F up-sampled matrix. If scale parameter is given negative
	 * null matrix is returned. Double scale factors are trimmed to integers
	 */
	public static DenseMatrix64F upsample(DenseMatrix64F matrix, int scale){
		if(scale<=0){
			return null;
		}
		DenseMatrix64F upscaled = new DenseMatrix64F(matrix.numRows*scale, matrix.numCols*scale);
		for(int i=0; i<upscaled.numRows; i++){
			for(int j=0; j<upscaled.numCols; j++){
				if((i%scale==0)&&(j%scale==0)){
					upscaled.unsafe_set(i,j,matrix.unsafe_get(i/scale,j/scale));
				} else{
					upscaled.unsafe_set(i,j,0);
				}
			}
		}
		return upscaled;
	}
	/**
	 * Function creates larger matrix by specified scale factor row wise. 
	 * New values are filled with zeros
	 * @param matrix DenseMatrix64F input matrix
	 * @param scale integer scale factor
	 * @return DenseMatrix64F up-sampled matrix
	 */
	public static DenseMatrix64F upsampleRows(DenseMatrix64F matrix, int scale){
		if(scale<=0){
			return null;
		}
		DenseMatrix64F upscaled = new DenseMatrix64F(matrix.numRows, matrix.numCols*scale);
		for(int i=0; i<upscaled.numRows; i++){
			for(int j=0; j<upscaled.numCols; j++){
				if(j%scale==0){
					upscaled.unsafe_set(i, j, matrix.unsafe_get(i, j/scale));
				} else{
					upscaled.unsafe_set(i, j, 0);
				}
			}
		}
		return upscaled;
	}
	
	/**
	 * Function creates larger matrix by specified scale factor row wise. 
	 * Allows specifying phase for which to offset data
	 * New values are filled with zeros
	 * @param matrix DenseMatrix64F input matrix
	 * @param scale integer scale factor
	 * @param offset integer specifying offset by which original data is offset
	 * must be in range (0 scale-1) otherwise null is returned
	 * @return DenseMatrix64F up-sampled matrix
	 */
	public static DenseMatrix64F upsampleRows(DenseMatrix64F matrix, int scale, int offset){
		if((scale<=0) ||(offset<0) ||(offset>=scale)){
			return null;
		}
		DenseMatrix64F upscaled = new DenseMatrix64F(matrix.numRows, matrix.numCols*scale);

		for(int i=0; i<upscaled.numRows; i++){
			for(int z=0; z<offset; z++){ 	// fill offset values
				upscaled.unsafe_set(i, z,0);
			}
			for(int j=0; j<upscaled.numCols; j++){
				if(j%scale==0){
					upscaled.unsafe_set(i, j+offset, matrix.unsafe_get(i, j/scale));
				} else{
					if(j+offset<upscaled.numCols){
					upscaled.unsafe_set(i, j+offset, 0);
					}
				}
			}
		}
		return upscaled;
	}
	
	/**
	 * Function creates larger matrix by specified scale factor column wise.
	 * Allows specifying phase by which to offset data 
	 * New values are filled with zeros
	 * @param matrix DenseMatrix64F input matrix
	 * @param scale integer scale factor
	 * @param offset integer specifying offset by which original data if offset
	 * must be in range (0, scale -1) otherwise null is returned
	 * @return DenseMatrix64F up-sampled matrix
	 */
	public static DenseMatrix64F upsampleCols(DenseMatrix64F matrix, int scale, int offset){
		if((scale<=0) ||(offset<0) ||(offset>=scale)){
			return null;
		}
		DenseMatrix64F upscaled = new DenseMatrix64F(matrix.numRows*scale, matrix.numCols);
		for(int i=0; i<upscaled.numRows; i++){
			for(int z=0; z<offset; z++){	// first set offset value with zeros
				for(int k=0; k<upscaled.numCols; k++){
					upscaled.unsafe_set(z, k, 0);
				}
			}
			for(int j=0; j<upscaled.numCols; j++){
				if(i%scale==0){
					upscaled.unsafe_set(i+offset, j, matrix.unsafe_get(i/scale, j));
				} else{
					if((i+offset)<upscaled.numRows){
						upscaled.unsafe_set(i+offset, j, 0);
					}
				}
			}
		}
		return upscaled;
	}
	/**
	 * Function creates larger matrix by specified scale factor column wise. 
	 * New values are filled with zeros
	 * @param matrix DenseMatrix64F input matrix
	 * @param scale integer scale factor
	 * @return DenseMatrix64F up-sampled matrix
	 */
	public static DenseMatrix64F upsampleCols(DenseMatrix64F matrix, int scale){
		if(scale<=0){
			return null;
		}
		DenseMatrix64F upscaled = new DenseMatrix64F(matrix.numRows*scale, matrix.numCols);
		for(int i=0; i<upscaled.numRows; i++){
			for(int j=0; j<upscaled.numCols; j++){
				if(i%scale==0){
					upscaled.unsafe_set(i, j, matrix.unsafe_get(i/scale, j));
				} else{
					upscaled.unsafe_set(i, j, 0);
				}
			}
		}
		return upscaled;
	}
	
	/**
	 * Function for bicubic convolution kernel value generation
	 * From paper: R. Keys, (1981). "Cubic convolution interpolation 
	 * for digital image processing". IEEE Transactions on Acoustics, 
	 * Speech, and Signal Processing. The kernel function is defined
	 * with following equations:
	 * h(x)={(a+2)|x|^3-(a+3)|x|^2+1, for |x|<=1,
	 * 		{a|x|^3-5a|x|^2+8a|x|-4a}, for 1<|x|<2.
	 * 		{0, 					 otherwise}
	 * 
	 * @param x argument for kernel function.
	 * @param a parameter for bicubic function modification
	 * @return double kernel value for given x
	 */
	public static double biCubic(double x, double a){
		double absx=Math.abs(x);
		double absx2=Math.pow(absx, 2);
		double absx3=Math.pow(absx, 3);
		if(absx<=1){
			return (a+2)*absx3-(a+3)*absx2+1;
		}
		if(absx<2){
			return a*absx3-5*a*absx2+8*a*absx-4*a;
		}
		return 0;
	}
	/**
	 * Function constructs interpolation kernel
	 * @param scale scale parameter
	 * @return scaled version of original matrix
	 */
	public static DenseMatrix64F generateBicubKernel(int scale){
		DenseMatrix64F kernel = new DenseMatrix64F(4*scale-1,1); //size of the kernel depends on scaling factor
																 // always odd number
		double scalefactor=0;
		int lowLim=-(kernel.getNumElements()-1)/2;
		int highLim=(kernel.getNumElements()-1)/2;
		for(int i=lowLim; i<=highLim;i++){
			scalefactor=1.0/scale;
			kernel.set(i-lowLim, biCubic(i*scalefactor,-0.5));
		}
		return kernel;
	}
	
	/** Function for image resizing. Currently performs Bicubic Convolution interpolation, and
	 * Proides only integer scale parameters greater that one
	 */
	public static DenseMatrix64F imResize(DenseMatrix64F matrix, int scaleFactor){
		DenseMatrix64F rowInterp = upsampleRows(matrix,scaleFactor);
		DenseMatrix64F kernel = generateBicubKernel(scaleFactor);
		// convolving rows
		for(int i=0; i<rowInterp.numRows; i++){
			CommonOps.insert(FilteringOperations.convolve1D(CommonOps.extract(rowInterp, i,i+1, 0, rowInterp.numCols), kernel), rowInterp, i, 0);
		}
		DenseMatrix64F colInterp = upsampleCols(rowInterp, scaleFactor);
		for(int i=0; i<colInterp.numCols; i++){
			CommonOps.insert(FilteringOperations.convolve1D(CommonOps.extract(colInterp, 0,colInterp.numRows, i, i+1), kernel), colInterp, 0, i);
		}
		return colInterp;
	}
}
