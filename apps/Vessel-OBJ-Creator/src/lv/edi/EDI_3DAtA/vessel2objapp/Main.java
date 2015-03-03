package lv.edi.EDI_3DAtA.vessel2objapp;
	
import org.ejml.data.DenseMatrix64F;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.stage.Stage;
import javafx.scene.Camera;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.TriangleMesh;
import lv.edi.EDI_3DAtA.common.VolumetricData;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.marchingcubes.TriangleMeshData;


public class Main extends Application {
	static MetaImage selectedTomographyScan;
	static MetaImage tomographyScanLungMasks;
	static VolumetricData volumeVesselSegmentationData;
	static int[] segmentatedDataRange = new int[2];
	static int selectedLayer=100;
	static DenseMatrix64F currentLayerImage;
	static TriangleMeshData trMeshData;
	static MeshView bloodVessel3DView;
	static TriangleMesh trMesh;
	static Camera camera3D;
	static float cameraRotAngleY=0;
	static float cameraRotAngleX=0;
	static float translateZ=-1000;

	@Override
	public void start(Stage primaryStage) {
		try {
			FXMLLoader loader = new FXMLLoader(getClass().getResource("MainScene.fxml"));
			Parent root = loader.load();
			AppController controller = loader.getController();
			controller.setMainStage(primaryStage);
			Scene scene = new Scene(root);
			primaryStage.setResizable(false);
			primaryStage.setTitle("CT Scan blood vessel extraction tool");
//			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			primaryStage.setScene(scene);
			primaryStage.show();
			 
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		launch(args);
	}
}
