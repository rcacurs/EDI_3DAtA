package lv.edi.EDI_3DAtA.vessel2objapp;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.MenuItem;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

public class AppController implements Initializable{
	private Stage mainStage;
	
	@FXML
	private MenuItem menuItemOpenCTFile;
	@FXML
    private ImageView ctScanImageView;
	@Override
	public void initialize(URL location, ResourceBundle resources) {
	}
	public void setMainStage(Stage stage){
		this.mainStage = stage;
	};
	
	
	
	@FXML
	public void selectCTScanFile(ActionEvent event){
		System.out.println("Clicked");
		System.out.println(menuItemOpenCTFile);
		FileChooser fileChooser = new FileChooser();
		ExtensionFilter extFilter = new ExtensionFilter("Meta Image files (*.mhd)", "*.mhd");
		fileChooser.getExtensionFilters().add(extFilter);
		fileChooser.setTitle("Open CT Scan File");
		File scanFile = fileChooser.showOpenDialog(mainStage);
		if(scanFile==null){
			((MenuItem)event.getSource()).getParentMenu().hide();
			return;
		}
		try {
			Main.selectedTomographyScan= new MetaImage(scanFile);
			Main.currentLayerImage=Main.selectedTomographyScan.getLayerImage(Main.selectedLayer);
			BufferedImage bImage = ImageVisualization.convDenseMatrixToBufImage(Main.currentLayerImage);
			WritableImage image = new WritableImage(bImage.getWidth(), bImage.getHeight());
			SwingFXUtils.toFXImage(bImage, image);
			ctScanImageView.setImage(image);
		} catch (IOException e) {
			System.out.println("Problem Loadin Specified File");
		}
		
		((MenuItem)event.getSource()).getParentMenu().hide();
	}
}
