package lv.edi.EDI_3DAtAtest.featureextractortest;

import java.io.IOException;

import javax.swing.JFrame;

import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

import org.ejml.data.DenseMatrix64F;

public class Main {
	static MetaImage metaImage;
	static DenseMatrix64F layerImage;
	static int selectedLayerIndex;
	static JFrame inputImageFrame = new JFrame("Input Image Frame");
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
		layerImage = metaImage.getLayerImage(selectedLayerIndex);
		// SHOW LAYER
		ImageVisualization.imshow(layerImage, inputImageFrame);
	}

}
