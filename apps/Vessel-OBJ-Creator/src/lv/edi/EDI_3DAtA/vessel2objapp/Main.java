package lv.edi.EDI_3DAtA.vessel2objapp;
	
import org.ejml.data.DenseMatrix64F;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.stage.Stage;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import lv.edi.EDI_3DAtA.imageio.MetaImage;


public class Main extends Application {
	static MetaImage selectedTomographyScan;
	static MetaImage tomographyScanLungMasks;
	static int selectedLayer=100;
	static DenseMatrix64F currentLayerImage;
	
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
