package lv.edi.EDI_3DAtA.bloodvesselsegm;

import org.ejml.data.DenseMatrix64F;

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
		DenseMatrix64F kernel = new DenseMatrix64F(3*scale+1,1); //size of the kernel depends on scaling factor
		double scalefactor=0;
		int lowLim=-kernel.getNumElements()/2;
		int highLim=(kernel.getNumElements()-1)/2;
		for(int i=lowLim; i<=highLim;i++){
			scalefactor=1.0/scale;
			kernel.set(i-lowLim, biCubic(i*scalefactor,-0.5));
		}
		return kernel;
	}

}
