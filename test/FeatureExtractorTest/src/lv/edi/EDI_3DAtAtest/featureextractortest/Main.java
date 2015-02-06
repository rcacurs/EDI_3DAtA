package lv.edi.EDI_3DAtAtest.featureextractortest;

import java.io.IOException;

import javax.swing.JFrame;

import lv.edi.EDI_3DAtA.bloodvesselsegm.LayerSMFeatures;
import lv.edi.EDI_3DAtA.bloodvesselsegm.SMFeatureExtractor;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

import org.ejml.data.DenseMatrix64F;

public class Main {
	static MetaImage metaImage;
	static DenseMatrix64F layerImage;
	static int selectedLayerIndex;
	static SMFeatureExtractor featureExtractor;
	static LayerSMFeatures layerFeatures;
	static JFrame inputImageFrame = new JFrame("Input Image Frame");
	static long time1, time2;
	public static void main(String[] args) {
		
		if(args.length<2){
			System.out.println("Program expects following arguments:");
			System.out.println("\t arg0 - path to the tomography scan file");
			System.out.println("\t arg1 - layer index to process");
			return;
		}
		
		// READ META IMAGE
		try {
			metaImage = new MetaImage(args[0]);
		} catch (IOException e) {
			System.out.println("Problem opening specified .mhd file");
			return;
		}
		// READ LAYER DATA FROM IMAGE
		try {
			
			selectedLayerIndex = Integer.parseInt(args[1]);
		} catch (NumberFormatException e) {
			System.out.println("Problem parcing specified layer. Specified layer must be specified as number!");
			return;
		}
		System.out.println("Reading layer...");
		time1 = System.currentTimeMillis();
		layerImage = metaImage.getLayerImage(selectedLayerIndex);
		time2 = System.currentTimeMillis();
		System.out.println("Layer read time: "+(time2-time1)+" [ms]");
		// SHOW LAYER
		ImageVisualization.imshow(layerImage, inputImageFrame);
		// EXTRACT FEATURES
		System.out.println("Extracting Features... ");
		featureExtractor = new SMFeatureExtractor("../../modules/res/dCodes", "../../modules/res/dMean", 5, 6);
		time1 = System.currentTimeMillis();
		layerFeatures = featureExtractor.extractLayerFeatures(layerImage);
		time2 = System.currentTimeMillis();
		System.out.println("Feature extraction time: "+(time2-time1)+" [ms]");
		
		// VISUALIZING FEATURES
		JFrame[] featureFrames = new JFrame[args.length-2];
		for(int i=0; i<args.length-2; i++){
			int featureNum;
			try {
				featureNum=Integer.parseInt(args[2+i]);
			} catch (NumberFormatException e) {
				System.out.println("Error: Feature indexes must be specified as numbers!");
				return;
			}
			DenseMatrix64F feature0 = layerFeatures.getFeature(featureNum);
			featureFrames[i] = new JFrame("Feature "+featureNum+" frame");
			ImageVisualization.imshow(feature0, featureFrames[i]);
		}
		
	}

}
