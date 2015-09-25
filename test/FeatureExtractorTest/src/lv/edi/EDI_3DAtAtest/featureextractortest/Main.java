package lv.edi.EDI_3DAtAtest.featureextractortest;

import java.io.IOException;

import javax.swing.JFrame;

import lv.edi.EDI_3DAtA.bloodvesselsegm.FilteringOperations;
import lv.edi.EDI_3DAtA.bloodvesselsegm.LayerSMFeatures;
import lv.edi.EDI_3DAtA.bloodvesselsegm.SMFeatureExtractor;
import lv.edi.EDI_3DAtA.common.DenseMatrixConversions;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

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
		int z=0;
		while(z<1){
			z++;
			try {
				
				selectedLayerIndex = Integer.parseInt(args[1]);
			} catch (NumberFormatException e) {
				System.out.println("Problem parcing specified layer. Specified layer must be specified as number!");
				return;
			}

//			System.out.println("Reading layer...");
			time1 = System.currentTimeMillis();
			layerImage = metaImage.getLayerImage(selectedLayerIndex);
			time2 = System.currentTimeMillis();
//			System.out.println("Layer read time: "+(time2-time1)+" [ms]");
			// SHOW LAYER
//			ImageVisualization.imshow(layerImage, inputImageFrame);
			// EXTRACT FEATURES
//			System.out.println("Extracting Features... ");
			DenseMatrix64F readF;
			DenseMatrixConversions.loadCSVtoDenseMatrix("C:\\Users\\Richards\\Dropbox\\EDI\\Projekts_3DAtA\\Eclipse-Workspace-Win\\TestFeatureExtractor\\dCodes");
			System.out.println("test");
			featureExtractor = new SMFeatureExtractor("C:\\Users\\Richards\\Dropbox\\EDI\\Projekts_3DAtA\\Eclipse-Workspace-Win\\TestFeatureExtractor\\dCodes", 
					                                   "C:\\Users\\Richards\\Dropbox\\EDI\\Projekts_3DAtA\\Eclipse-Workspace-Win\\TestFeatureExtractor\\dMean", 5, 6);
			time1 = System.currentTimeMillis();
			layerFeatures = featureExtractor.extractLayerFeatures(layerImage);
			time2 = System.currentTimeMillis();
			
			DenseMatrix64F featureMat = layerFeatures.getFeatures();
			CommonOps.transpose(featureMat);
			//DenseMatrixConversions.saveDenseMatrixToCSV(featureMat, "featuresMat");
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

}
