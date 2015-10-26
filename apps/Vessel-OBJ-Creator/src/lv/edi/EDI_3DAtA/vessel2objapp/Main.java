package lv.edi.EDI_3DAtA.vessel2objapp;
	
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import javafx.application.Application;
import javafx.application.HostServices;
import javafx.fxml.FXMLLoader;
import javafx.stage.Stage;
import javafx.scene.Camera;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.TriangleMesh;
import lv.edi.EDI_3DAtA.bloodvesselsegm.SMFeatureExtractor;
import lv.edi.EDI_3DAtA.common.DenseMatrixConversions;
import lv.edi.EDI_3DAtA.common.VolumetricData;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.marchingcubes.TriangleMeshData;
import lv.edi.EDI_3DAtA.opencvcudainterface.Compute;


public class Main extends Application {
	static MetaImage selectedTomographyScan;
	static MetaImage tomographyScanLungMasks;
	static VolumetricData volumeVesselSegmentationData;
	static int[] segmentatedDataRange = new int[2];
	static int selectedLayer=100;
	static DenseMatrix64F means;
	static DenseMatrix64F codes;
	static DenseMatrix64F codesTr;
	static DenseMatrix64F model;
	static DenseMatrix64F scaleParamsMean;
	static DenseMatrix64F scaleParamsSd;
	static DenseMatrix64F currentLayerImage;
	static TriangleMeshData trMeshData;
	static MeshView bloodVessel3DView;
	static TriangleMesh trMesh;
	static Camera camera3D;
	static float cameraRotAngleZ=0;
	static float cameraRotAngleX=0;
	static float translateZ=-1000;
	static Compute compute;
	public static HostServices hostServices;

	@Override
	public void start(Stage primaryStage) {
		try {
			System.out.println("Trying to load library");
			compute = new Compute(); // initialize opencv cuda interface
			System.out.println("Cuda computation library loaded! Blood vessel segmentation will be performed using GPU");
			//compute.test();
		} catch (UnsatisfiedLinkError e1) {
			// TODO Auto-generated catch block
			System.out.println("Library for computations using cuda not loaded!");
			System.out.println(e1.getMessage());
			compute=null;
		}
		codes = DenseMatrixConversions.loadCSVtoDenseMatrixFromInputStream(SMFeatureExtractor.class.getResourceAsStream("dCodes.csv"));
		CommonOps.transpose(codes);
		codesTr = DenseMatrixConversions.loadCSVtoDenseMatrixFromInputStream(SMFeatureExtractor.class.getResourceAsStream("dCodes.csv"));
		means = DenseMatrixConversions.loadCSVtoDenseMatrixFromInputStream(SMFeatureExtractor.class.getResourceAsStream("dMean.csv"));
		model = DenseMatrixConversions.loadCSVtoDenseMatrixFromInputStream(SMFeatureExtractor.class.getResourceAsStream("model.csv"));
		scaleParamsMean = DenseMatrixConversions.loadCSVtoDenseMatrixFromInputStream(SMFeatureExtractor.class.getResourceAsStream("scaleparamsMean.csv"));
		scaleParamsSd = DenseMatrixConversions.loadCSVtoDenseMatrixFromInputStream(SMFeatureExtractor.class.getResourceAsStream("scaleparamsSd.csv"));
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
		hostServices = getHostServices();
	}
	
	public static void main(String[] args) {
		launch(args);
	}
}
