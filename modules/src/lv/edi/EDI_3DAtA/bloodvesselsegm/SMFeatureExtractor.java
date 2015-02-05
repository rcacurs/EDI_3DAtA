package lv.edi.EDI_3DAtA.bloodvesselsegm;

import javax.swing.JFrame;

import lv.edi.EDI_3DAtA.common.DenseMatrixConversions;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

///**
// * 
// * @author Code - Ričards Cacurs.
// * 
// *  Class representing Stacked Multiscale Feature extractor from paper:
// *  Adam Coates and Andrew Y. Ng "Learning Feature Representations with K-means", 
// *  Neural Networks: Tricks of the Trade, 2nd edn, Springer, 2012.
// *
// */
public class SMFeatureExtractor {
	private DenseMatrix64F codes;
	private DenseMatrix64F means;
	private int patchSize;
	private int numScales;
	
	/**
	 * Feature extractor constructor
	 * @param codesFileName - file name to .csv file containing codes (omitting extension)
	 * @param meansFileName - file name to .csv file containing means (omitting extension)
	 * @param patchSize - patch size for algorithm
	 * @param numScales - number of scales
	 */
	public SMFeatureExtractor(String codesFileName, String meansFileName, int patchSize, int numScales){
		setCodes(codesFileName);	
		CommonOps.transpose(codes);
		setMeans(meansFileName);
		this.patchSize = patchSize;
		this.numScales = numScales;
	}

	/**
	 * Function for setting feature extractor codes
	 * @param fileName path to the .csv file containing codes(not including file extension).
	 */
	public void setCodes(String fileName){
		codes = DenseMatrixConversions.loadCSVtoDenseMatrix(fileName);
	}
	
	/**
	 * Function for setting feature extractor means
	 * @param fileName path to the .csv file containing codes(not including file extension)
	 */
	public void setMeans(String fileName){
		means = DenseMatrixConversions.loadCSVtoDenseMatrix(fileName);
	}
	
	/**
	 *  Sets number of scales used in algorithm
	 * @param numScales number of scales for feature extractor
	 */
	public void setScales(int numScales){
		this.numScales = numScales;
	}
	
	//TODO 	//UPDATE WITH RETURN OF LayerFeatures object
	/**
	 * Extracts Layer features
	 * @param layer layer image in Densematrix64F format for which to extract features
	 * 
	 */
	public void extractLayerFeatures(DenseMatrix64F layer){
		
		GaussianPyramid gPyramid = new GaussianPyramid(layer, numScales, 5, 1);
		DenseMatrix64F multiFilteredLayer;
		DenseMatrix64F filteredImage;
		DenseMatrix64F upscaledImage;
		for(int i=0; i<gPyramid.size(); i++){ // for each scale
			multiFilteredLayer = SMFilterBlock.filter(gPyramid.getLayer(i), patchSize, codes, means);
			for(int j=0; j<multiFilteredLayer.getNumCols(); j++){ // for each filter
				filteredImage=CommonOps.extract(multiFilteredLayer, 0, multiFilteredLayer.numRows, j, j+1);
				filteredImage.reshape(gPyramid.getLayer(i).numCols, gPyramid.getLayer(i).numRows);
				CommonOps.transpose(filteredImage); // because of the row major ordering
				if(i>0){
					upscaledImage = MatrixScaling.imResize(filteredImage, (int)Math.pow(2, i));
				} else{
					upscaledImage = filteredImage;

				}
				//TODO: fill features object
			}
		}
		
	}

}
