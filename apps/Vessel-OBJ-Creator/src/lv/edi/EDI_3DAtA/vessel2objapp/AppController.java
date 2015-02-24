package lv.edi.EDI_3DAtA.vessel2objapp;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

import javafx.concurrent.Task;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyEvent;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;
import lv.edi.EDI_3DAtA.bloodvesselsegm.SMFeatureExtractor;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

public class AppController implements Initializable{
	private Stage mainStage;
	
	private Task<Integer> activeSegmentationTask;
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
	@FXML
	private TextField textFieldSegmHighRange;
	@FXML
	private TextField textFieldSegmLowRange;
	@FXML
	private ToggleButton buttonSegmentBloodVessels;
	@FXML
	private ProgressIndicator progressIndicatorSegmentation;
	
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
			Main.selectedTomographyScan = new MetaImage(scanFile);
			
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
			
			File scanDir = Main.selectedTomographyScan.getElementHeaderFile().getParentFile().getParentFile();
			File scanLungMask = new File(scanDir.toString()+"\\Lungmasks\\"+scanFile.getName());
			Main.tomographyScanLungMasks = new MetaImage(scanLungMask);
			
			
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
	
	@FXML
	// calback for button that start blood vessel segmentation
		
	public void onButtonSegmentBloodVessels(ActionEvent event){
		if(buttonSegmentBloodVessels.isSelected()){
			if(Main.selectedTomographyScan==null){
				Alert alert = new Alert(AlertType.ERROR);
				alert.setTitle("Error");
				alert.setHeaderText("CT file not opened");
				alert.setContentText("Please open CT scan file: File->Open CT Scan File...");
				alert.showAndWait();
				buttonSegmentBloodVessels.setSelected(false);
				return;
			}
			if(Main.tomographyScanLungMasks==null){
				Alert alert = new Alert(AlertType.ERROR);
				alert.setTitle("Error");
				alert.setHeaderText("Problem finding lung mask file!");
				alert.setContentText("Please check if there to opened CT scan file is corresponding file in folder relative to it (../Lungscans)");
				alert.showAndWait();
				buttonSegmentBloodVessels.setSelected(false);
				return;
			}
			int[] layerRange = parseSelectedLayersRange();
			if(layerRange==null){
				buttonSegmentBloodVessels.setSelected(false);
				return;
			}
			// run background task
			
			// Vessel segmentation TASK
			Task<Integer> task = new Task<Integer>(){
			@Override
				protected Integer call(){
					int iterations;
					System.out.println("Task start");
					SMFeatureExtractor featureExtractor = new SMFeatureExtractor("dCodes", 
	                "dMean", 5, 6);
					for(iterations = layerRange[0]; iterations <=layerRange[1]; iterations++){
						if(isCancelled()){
							break;
						}
									
									
						updateProgress(iterations-layerRange[0], layerRange[1]+1-layerRange[0]);
					}
					return iterations;
				}
			};
			progressIndicatorSegmentation.progressProperty().bind(task.progressProperty());
			 task.setOnSucceeded(e -> {
				 	System.out.println("Tasks finished!");
				 	buttonSegmentBloodVessels.setSelected(false);
					activeSegmentationTask=null;
			    });
			new Thread(task).start();
			activeSegmentationTask = task;
		} else{
			System.out.println("Click");
			if(activeSegmentationTask!=null){
				activeSegmentationTask.cancel();
				activeSegmentationTask=null;
			}
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

	// method parses selected range for layers that are to be scanned.
	private int[] parseSelectedLayersRange(){
		int range[] = new int[2];
		Alert alert = new Alert(AlertType.ERROR);
		try {
			range[0] = Integer.parseInt(textFieldSegmLowRange.getText());
			range[1] = Integer.parseInt(textFieldSegmHighRange.getText());
		} catch (NumberFormatException e) {
			
			alert.setTitle("Error");
			alert.setHeaderText("Error parsing given layer range!");
			alert.setContentText("There seems to be problem with specified layer range for wich to perform segmentation.");
			alert.showAndWait();
			return range;
		}
		if(range[0]>range[1]){
			alert.setTitle("Error");
			alert.setHeaderText("Error parsing given layer range!");
			alert.setContentText("First specified range should be smaller index.");
			alert.showAndWait();
			return range;
		}
		if(range[1]>=((int)Main.selectedTomographyScan.getDimSize().get(2))){
			alert.setTitle("Error");
			alert.setHeaderText("Error parsing given layer range!");
			alert.setContentText("Selected layer range shouldn't exceed number of layers in selected CT scan.");
			alert.showAndWait();
			return range;
		}

		return range;
	}
	
}
