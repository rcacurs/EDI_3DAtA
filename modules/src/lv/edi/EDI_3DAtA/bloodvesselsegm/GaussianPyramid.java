package lv.edi.EDI_3DAtA.bloodvesselsegm;

import java.util.ArrayList;

import boofcv.alg.filter.convolve.GConvolveImageOps;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.struct.convolve.Kernel2D_F32;
import boofcv.struct.convolve.Kernel2D_I32;
import boofcv.struct.image.ImageFloat32;
import boofcv.struct.image.ImageUInt8;

/**
 * Class representing Gaussian pyramid of image. Each layer of pyramid is two times smaller
 * @author Ričards Cacurs
 *
 */
public class GaussianPyramid {
	private ArrayList<ImageFloat32> layers;
	
	/** Constructor that creates Gaussian pyramid with specified parameters.
	 * 
	 * @param src input image, gaussian pyramid layer 0
	 * @param numberOfLayers  number of layers for pyramid
	 * @param kernelWidth size for the Gaussian filter kernel in pixels
	 * @param kernelSigma sigma parameter for Gaussian filter
	 */
	public GaussianPyramid(ImageFloat32 src, int numberOfLayers, int kernelWidth, float kernelSigma){
		layers = new ArrayList<ImageFloat32>(numberOfLayers);
		Kernel2D_F32 kernel = FactoryKernelGaussian.gaussian2D_F32(1.0, (kernelWidth-1)/2, true);
		layers.add(src);
		for(int i=1; i<numberOfLayers; i++){
			ImageFloat32 output = new ImageFloat32(layers.get(i-1).width, layers.get(i-1).height);
			GConvolveImageOps.convolveNormalized(kernel, layers.get(i-1), output);
			ImageFloat32 outputDownsample = new ImageFloat32(layers.get(i-1).width/2, layers.get(i-1).height/2);
			for(int j=0; j<outputDownsample.height; j++){
				for(int k=0; k<outputDownsample.width; k++){
					outputDownsample.set(k, j, output.get(k*2, j*2));
				}
			}
			layers.add(outputDownsample);
		}
	}
	
	/**
	 * Function for accessing pyramid layer
	 * @param layer layer index
	 * @return ImageFloat32 returns specific layer. Returns null if index 
	 * larger that number of pyramid layers
	 */
	public ImageFloat32 getLayer(int layer){
		if(layer<layers.size() && layer>=0){
			return layers.get(layer);
		} else{
			return null;
		}
	}
	
	/**
	 *  Returns number of layers in Gaussian pyramid
	 * @return int Layer count in Gaussian pyramid
	 */
	public int size(){
		return layers.size();
	}
}
