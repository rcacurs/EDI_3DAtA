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
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyEvent;
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
	@FXML
	private TextField textFieldSelectedLayerIdx;
	@FXML
	private Button btnNavigateLayersUp;
	@FXML
	private Button btnNavigateLayersDown;
	@FXML
	private Label fieldScanFilePath;
	@FXML
	private Label labelScanDimX;
	@FXML
	private Label labelScanDimY;
	@FXML
	private Label labelScanDimZ;
	@FXML
	private Label labelScanSpacingX;
	@FXML
	private Label labelScanSpacingY;
	@FXML
	private Label labelScanSpacingZ;
	
	@Override
	public void initialize(URL location, ResourceBundle resources) {
	}
	public void setMainStage(Stage stage){
		this.mainStage = stage;
	};
	
	@FXML 
	void onTextFieldSelectedLayer(ActionEvent event){
		int index;
		try {
			index=Integer.parseInt(textFieldSelectedLayerIdx.getText());
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			index=0;
		}
		updateSelectedLayerIndex(index);
		updateSelectefLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
		
	}
	@FXML 
	public void onBtnNavigateLayerUp(){
		int index = Integer.parseInt(textFieldSelectedLayerIdx.getText());
		updateSelectedLayerIndex(index+1);
		updateSelectefLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
	}
	@FXML
	public void onBtnNavigateLayerDown(){
		int index = Integer.parseInt(textFieldSelectedLayerIdx.getText());
		updateSelectedLayerIndex(index-1);
		updateSelectefLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
	}
	@FXML
	public void selectCTScanFile(ActionEvent event){
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
			updateSelectedLayerIndex(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
			updateSelectefLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
			
			//update selected scan file field
			fieldScanFilePath.setText(scanFile.toString());
			//update scan dimmensions fields
			labelScanDimX.setText(""+(int)((Main.selectedTomographyScan.getDimSize().get(0))));
			labelScanDimY.setText(""+(int)((Main.selectedTomographyScan.getDimSize().get(1))));
			labelScanDimZ.setText(""+(int)((Main.selectedTomographyScan.getDimSize().get(2))));
			//update selected scan element spacing
			labelScanSpacingX.setText(String.format("%.3f",Main.selectedTomographyScan.getElementSpacing().get(0)));
			labelScanSpacingY.setText(String.format("%.3f",Main.selectedTomographyScan.getElementSpacing().get(1)));
			labelScanSpacingZ.setText(String.format("%.3f",Main.selectedTomographyScan.getElementSpacing().get(2)));
			
			
		} catch (IOException e) {
			System.out.println("Problem Loadin Specified File");
		}
		
		((MenuItem)event.getSource()).getParentMenu().hide();
	}
	@FXML
	public void filterLayerSelecteTxtField(KeyEvent arg0){
		String chara = arg0.getCharacter();
		if("0123456789".contains(chara)){
		} else{
			arg0.consume();
		}
	}
	
	// HELPER FUNCTIONS
	
	// updates selected layer indes in layer navigation panel
	private void updateSelectedLayerIndex(int newIndex){
		if(newIndex>=0){
			if(Main.selectedTomographyScan!=null){
				if(newIndex>=Main.selectedTomographyScan.getDimSize().get(2)){
					textFieldSelectedLayerIdx.setText(Integer.toString((int)Main.selectedTomographyScan.getDimSize().get(2)-1));
				} else{
					textFieldSelectedLayerIdx.setText(Integer.toString(newIndex));
				}
			} else{
				textFieldSelectedLayerIdx.setText(Integer.toString(newIndex));
			}
			
		};
	}
	// updates view with new image if file is opened
	private void updateSelectefLayerImage(int layerIndex){
		if(Main.selectedTomographyScan!=null){
			Main.currentLayerImage=Main.selectedTomographyScan.getLayerImage(layerIndex);
			BufferedImage bImage = ImageVisualization.convDenseMatrixToBufImage(Main.currentLayerImage);
			WritableImage image = new WritableImage(bImage.getWidth(), bImage.getHeight());
			SwingFXUtils.toFXImage(bImage, image);
			ctScanImageView.setImage(image);
		}
	}
}
