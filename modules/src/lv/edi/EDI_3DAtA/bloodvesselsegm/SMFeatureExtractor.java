package lv.edi.EDI_3DAtA.bloodvesselsegm;

import java.net.URL;

import lv.edi.EDI_3DAtA.common.DenseMatrixConversions;

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
	
	
	/** Default constructor for SMFeatureExtractor (CUrrentlu not working :))
	 * it sets default dMeans.csv and dCodes.csv /res file path, relative from class path
	 * set patch size 5, and number of scales 6
	 */
	public SMFeatureExtractor(){
		URL urlCodes = SMFeatureExtractor.class.getResource("dCodes.csv");
		URL urlMeans = SMFeatureExtractor.class.getResource("dMean.csv");
		System.out.println(urlCodes);
		System.out.println(urlMeans);
		String strCodes = urlCodes.getPath();
		String strMeans = urlMeans.getPath();
		System.out.println(strCodes);
		System.out.println(strMeans);
		setCodes(strCodes.substring(0, strCodes.length()-4));
		CommonOps.transpose(codes);
		setMeans(strMeans.substring(0, strMeans.length()-4));
		this.patchSize=5;
		this.numScales=6;
	}
	
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
		System.out.println("Set codes");
		System.out.println(fileName);
		codes = DenseMatrixConversions.loadCSVtoDenseMatrix(fileName);
	}
	
	/**
	 * Function for setting feature extractor means
	 * @param fileName path to the .csv file containing codes(not including file extension)
	 */
	public void setMeans(String fileName){
		System.out.println(fileName);
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
	 * @return LayerSMFeatures features of image
	 * 
	 */
	public LayerSMFeatures extractLayerFeatures(DenseMatrix64F layer){
		LayerSMFeatures features = new LayerSMFeatures(layer.numRows, layer.numCols, numScales*codes.numCols+1);
		GaussianPyramid gPyramid = new GaussianPyramid(layer, numScales, 5, 1);
		DenseMatrix64F multiFilteredLayer;
		DenseMatrix64F filteredImage;
		DenseMatrix64F upscaledImage;
		int featureIndex=0;
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
				features.setFeature(featureIndex, upscaledImage.data);
				featureIndex++;
			}
		}
		return features;
		
	}

}
