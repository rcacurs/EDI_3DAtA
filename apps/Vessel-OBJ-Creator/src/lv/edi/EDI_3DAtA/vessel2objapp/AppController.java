package lv.edi.EDI_3DAtA.vessel2objapp;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.ResourceBundle;

import org.ejml.data.DenseMatrix64F;

import javafx.concurrent.Task;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.AmbientLight;
import javafx.scene.Camera;
import javafx.scene.Group;
import javafx.scene.PerspectiveCamera;
import javafx.scene.SceneAntialiasing;
import javafx.scene.SubScene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.Tab;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;
import javafx.scene.shape.CullFace;
import javafx.scene.shape.DrawMode;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.TriangleMesh;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Translate;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;
import lv.edi.EDI_3DAtA.bloodvesselsegm.LayerSMFeatures;
import lv.edi.EDI_3DAtA.bloodvesselsegm.SMFeatureExtractor;
import lv.edi.EDI_3DAtA.bloodvesselsegm.SoftmaxRegrClassifier;
import lv.edi.EDI_3DAtA.common.VolumetricData;
import lv.edi.EDI_3DAtA.imageio.MetaImage;
import lv.edi.EDI_3DAtA.marchingcubes.MarchingCubes;
import lv.edi.EDI_3DAtA.visualization.ImageVisualization;

public class AppController implements Initializable{
	private Stage mainStage;
	
	private Task<Integer> activeSegmentationTask;
	@FXML
	private MenuItem menuItemOpenCTFile;
	@FXML
	private MenuItem menuExportToObj;
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
	@FXML
	private TextField textFieldSegmentationThreshold;
	@FXML
	private CheckBox cbShowSegmentation;
	@FXML
	private ToggleButton buttonGenerate3D;
	@FXML
	private ProgressIndicator progressIndicator3D;
	@FXML
	private Group group3D;
	@FXML
	private VBox box3DLayout;
	@FXML
	private Tab tab3D;
	// variables used for mouse events
	double previousX=0;
	double previousY=0;
	double rotationSensitivity=0.1;
	
	@Override
	public void initialize(URL location, ResourceBundle resources) {
		
		Box box = new Box(5, 5, 5);
		box.setMaterial(new PhongMaterial(Color.RED));
		box.setCullFace(CullFace.NONE);
		Group root = new Group();
		Main.bloodVessel3DView = new MeshView();
		float[] points = {10, 10, 10,
				  		  20, 10, 10, 
				  		  15, 20, 10};
		float[] texCoords = {0, 0};
		int[] faces = {0, 0, 1, 0, 2, 0};
		
		Main.trMesh = new TriangleMesh();
//		trMesh2.getPoints().setAll(points);
//		trMesh2.getTexCoords().setAll(texCoords);
//		trMesh2.getFaces().setAll(faces);
		
		
		
		//Main.bloodVessel3DView.setMesh(trMesh2);
		Main.bloodVessel3DView.setCullFace(CullFace.NONE);
		Main.bloodVessel3DView.setDrawMode(DrawMode.FILL);
		Main.bloodVessel3DView.setMaterial(new PhongMaterial(Color.RED));
		AmbientLight ambLight = new AmbientLight();
		Main.camera3D = new PerspectiveCamera(true);
		Main.camera3D.setFarClip(5000);
		root.getChildren().add(Main.bloodVessel3DView);
		root.getChildren().add(ambLight);
		System.out.println("box size"+box3DLayout.getHeight()+box3DLayout.getWidth());
		SubScene subScene = new SubScene(root, box3DLayout.getHeight(), box3DLayout.getWidth(), true, SceneAntialiasing.BALANCED);
		subScene.setCamera(Main.camera3D);
		root.getChildren().add(Main.camera3D);
		
		Main.camera3D.getTransforms().addAll(new Rotate(0, Rotate.Y_AXIS), new Rotate(0, Rotate.X_AXIS),
                new Translate(0, 0, Main.translateZ));
		

		//camera.getTransforms().setAll(new Rotate(1, Rotate.Z_AXIS));
		subScene.heightProperty().bind(box3DLayout.heightProperty());
		subScene.widthProperty().bind(box3DLayout.widthProperty());
		
		
		
		group3D.getChildren().add(subScene);
		
		float[] points2 = {-10, 10, 0,
				  0, -10, 0,
				  10, 10, 0};
		Main.trMesh.getPoints().setAll(points2);
		Main.trMesh.getTexCoords().setAll(texCoords);
		Main.trMesh.getFaces().setAll(faces);
		Main.bloodVessel3DView.setMesh(Main.trMesh);
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
		updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
		
	}
	@FXML 
	public void onBtnNavigateLayerUp(){
		int index = Integer.parseInt(textFieldSelectedLayerIdx.getText());
		updateSelectedLayerIndex(index+1);
		updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
	}
	@FXML
	public void onBtnNavigateLayerDown(){
		int index = Integer.parseInt(textFieldSelectedLayerIdx.getText());
		updateSelectedLayerIndex(index-1);
		updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
	}
	@FXML
	public void onSegmentationThresholdChange(ActionEvent event){
		updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
	}
	@FXML
	public void selectCTScanFile(ActionEvent event){
		FileChooser fileChooser = new FileChooser();
		ExtensionFilter extFilter = new ExtensionFilter("Meta Image files (*.mhd)", "*.mhd");
		fileChooser.getExtensionFilters().add(extFilter);
		fileChooser.setTitle("Open CT Scan File");
		File scanFile = fileChooser.showOpenDialog(mainStage);
		if(scanFile==null){
			System.out.println("ScanFile null");
			((MenuItem)event.getSource()).getParentMenu().hide();
			return;
		}
		try {
			Main.selectedTomographyScan = new MetaImage(scanFile);
			Main.volumeVesselSegmentationData = null;
			updateSelectedLayerIndex(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
			updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
			
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
	public void exportToObj(ActionEvent event){
		if(Main.volumeVesselSegmentationData==null){
			Alert alert = new Alert(AlertType.ERROR);
			alert.setTitle("Error");
			alert.setHeaderText("Blood vessel segmentation data not available");
			alert.setContentText("Please first run blood bessel segmentation!");
			alert.showAndWait();
			return;
		}
		double threshold;
		try{
			threshold = Double.parseDouble(textFieldSegmentationThreshold.getText());
		}catch(NumberFormatException ex){
			Alert alert = new Alert(AlertType.ERROR);
			alert.setTitle("Error");
			alert.setHeaderText("Specified threshold value wrong");
			alert.setContentText("Please check entered threshold value and ensure it is in range from 0.0 to 1.0!");
			alert.showAndWait();
			return;
		}
		if(threshold>1){
			threshold = 1;
		}
		DirectoryChooser dirChooser = new DirectoryChooser();
		dirChooser.setTitle("Select output folder for .obj file");
		File exportDir = dirChooser.showDialog(mainStage);
		if(exportDir==null){
			return;
		}
		Alert alert = new Alert(AlertType.INFORMATION);
		alert.setTitle("Exporting to .obj");
		alert.setHeaderText(null);
		alert.setContentText("Exporting to .obj file...");
		alert.show();
		
		MarchingCubes mc = new MarchingCubes(Main.volumeVesselSegmentationData);
		//ArrayList<DenseMatrix64F> vessel3DVertexes = mc.generateIsoSurface(threshold);
		
//		try {
//			Date date = new Date();
//			GregorianCalendar cal = new GregorianCalendar();
//			cal.setTime(date);
//			//MarchingCubes.saveVerticesToObj(vessel3DVertexes, exportDir.toString()+"/blood-vessel-model-"+(cal.get(Calendar.DAY_OF_MONTH))+""
////																						   +(cal.get(Calendar.MONTH)+1)+""
////																						   +(cal.get(Calendar.YEAR))+""
////																						   +(cal.get(Calendar.HOUR))+""+
////																						   +(cal.get(Calendar.MINUTE))+".obj");
//		} catch (FileNotFoundException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
////		vessel3DVertexes = null;
		alert.close();
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
	public void filterThresholdSelectedTxtField(KeyEvent arg0){
		String chara = arg0.getCharacter();
		if("0123456789.".contains(chara)){	
		} else{
			arg0.consume();
		}
	}
	// callback for check-box for blood vessel segmentation visualisation
	@FXML
	public void onCBShowSegmentation(ActionEvent event){
		updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
	}
	
	// calback for button that start blood vessel segmentation
	@FXML	
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
					int layer;
					System.out.println("Task start");
					SMFeatureExtractor featureExtractor = new SMFeatureExtractor();
					
					SoftmaxRegrClassifier classifier = new SoftmaxRegrClassifier(Main.selectedTomographyScan.getLayerImage(0).numRows, Main.selectedTomographyScan.getLayerImage(0).numCols);
					DenseMatrix64F layerImage;
					DenseMatrix64F layerMask;
					DenseMatrix64F layerVesselSegmentated;
					VolumetricData segmentationDataLocal = new VolumetricData();
					LayerSMFeatures layerFeatures;
					for(layer = layerRange[0]; layer <=layerRange[1]; layer++){
						if(isCancelled()){
							updateProgress(0, 100);
							break;
						}
						
						layerImage = Main.selectedTomographyScan.getLayerImage(layer);	
						layerFeatures = featureExtractor.extractLayerFeatures(layerImage);
						layerMask = Main.tomographyScanLungMasks.getLayerImage(layer);
						
						classifier.setData(layerFeatures);
						classifier.setMaskImage(layerMask);
						
						classifier.classify();
						
						layerVesselSegmentated = classifier.getResult();
						segmentationDataLocal.addLayer(layerVesselSegmentated);
						
						updateProgress(layer-layerRange[0]+1, (layerRange[1]-layerRange[0])+1);
						layerFeatures=null;
						layerMask=null;
						layerImage=null;
					}
					Main.volumeVesselSegmentationData=segmentationDataLocal;
					segmentationDataLocal=null;
					Main.segmentatedDataRange=layerRange;
					return 100;
				}
			};
			progressIndicatorSegmentation.progressProperty().bind(task.progressProperty());
			 task.setOnSucceeded(e -> {
				 	System.out.println("Tasks finished!");
				 	updateSelectedLayerImage(Integer.parseInt(textFieldSelectedLayerIdx.getText()));
				 	buttonSegmentBloodVessels.setSelected(false);
					//activeSegmentationTask=null;
					
			    });
			 task.setOnCancelled(e -> {
				 	progressIndicatorSegmentation.progressProperty().unbind();
				 	progressIndicatorSegmentation.setProgress(0);
				 	buttonSegmentBloodVessels.setSelected(false);
					//activeSegmentationTask=null;
					
			    });
			Thread thr = new Thread(task);
			thr.setPriority(Thread.MAX_PRIORITY);
			thr.start();
			activeSegmentationTask = task;
		} else{
			System.out.println("Click");
			if(activeSegmentationTask!=null){
				activeSegmentationTask.cancel();
				activeSegmentationTask=null;
			}
		}
		
	}
	
	@FXML
	public void onButtonGenerate3D(ActionEvent event){
		if(buttonGenerate3D.isSelected()){
			if(Main.volumeVesselSegmentationData!=null){
				
				// run background task
				// Vessel segmentation TASK
				double threshold;
				try{
					threshold = Double.parseDouble(textFieldSegmentationThreshold.getText());
				}catch(NumberFormatException ex){
					Alert alert = new Alert(AlertType.ERROR);
					alert.setTitle("Error");
					alert.setHeaderText("Specified threshold value wrong");
					alert.setContentText("Please check entered threshold value and ensure it is in range from 0.0 to 1.0!");
					alert.showAndWait();
					return;
				}
				if(threshold>1){
					threshold = 1;
				}
				Task<Integer> task = new Task<Integer>(){
				
				@Override
					protected Integer call(){
						System.out.println("Task start");
						double threshold;
						try{
							threshold = Double.parseDouble(textFieldSegmentationThreshold.getText());
						}catch(NumberFormatException ex){
							Alert alert = new Alert(AlertType.ERROR);
							alert.setTitle("Error");
							alert.setHeaderText("Specified threshold value wrong");
							alert.setContentText("Please check entered threshold value and ensure it is in range from 0.0 to 1.0!");
							alert.showAndWait();
							return 0;
						}
						if(threshold>1){
							threshold = 1;
						}
						MarchingCubes mc = new MarchingCubes(Main.volumeVesselSegmentationData);
						mc.getProgressProperty().addListener((obs, oldProgress, newProgress) ->{
					    	updateProgress((double)newProgress, 1.0);
					    	System.out.println("Update Progress!");
					    	}
					    );
						
						Main.trMeshData = mc.generateIsoSurface(threshold);
						System.out.println("Surface generated!");
						return 100;
					}
				};
				progressIndicator3D.progressProperty().bind(task.progressProperty());
				 task.setOnSucceeded(e -> {
					 	System.out.println("Tasks finished, Succeeded!");
					 	buttonGenerate3D.setSelected(false);
						
					 	System.out.println("vert"+Main.trMeshData.vertices+" tex: "+Main.trMeshData.texCoords+" faces: "+Main.trMeshData.faces);
					 	System.out.println("Num vertices: "+Main.trMeshData.vertices.length+" Num tex:"+Main.trMeshData.texCoords.length+" numfaces: "+Main.trMeshData.faces.length);
						Main.trMesh.getPoints().setAll(Main.trMeshData.vertices);
						Main.trMesh.getTexCoords().setAll(Main.trMeshData.texCoords);
						Main.trMesh.getFaces().setAll(Main.trMeshData.faces);
				
						Main.bloodVessel3DView.setMesh(Main.trMesh);
						Main.bloodVessel3DView.setMaterial(new PhongMaterial(Color.RED));
						Main.bloodVessel3DView.setDrawMode(DrawMode.FILL);
						Main.bloodVessel3DView.setCullFace(CullFace.NONE);
						Main.bloodVessel3DView.getTransforms().add(new Translate(-Main.trMeshData.center[0], -Main.trMeshData.center[1], -Main.trMeshData.center[2]));
						System.out.println("Surface changed");
						
				    });
				 task.setOnCancelled(e -> {
					 	progressIndicator3D.progressProperty().unbind();
					 	progressIndicator3D.setProgress(0);
					 	buttonGenerate3D.setSelected(false);
						//activeSegmentationTask=null;
						
				    });
				Thread thr = new Thread(task);
				thr.setPriority(Thread.MAX_PRIORITY);
				thr.start();
				activeSegmentationTask = task;
			} else{
				Alert alert = new Alert(AlertType.ERROR);
				alert.setTitle("Error");
				alert.setHeaderText("Blood vessel segmentation data no avialable!");
				alert.setContentText("Please run blood vessel segmentation first!");
				alert.showAndWait();
				buttonGenerate3D.setSelected(false);
			}
		} else{
			
		}
	}
	@FXML
	public void on3DViewDrag(MouseEvent event){
		double currentX=event.getX();
		double currentY=event.getY();
		double difX=previousX-currentX;
		double difY=previousY-currentY;
		
		System.out.println("Drag event!");
		Main.cameraRotAngleY+=difX*rotationSensitivity;
		Main.cameraRotAngleX+=difY*rotationSensitivity;
        Main.camera3D.getTransforms().set(0, new Rotate(Main.cameraRotAngleY, Rotate.Y_AXIS));
        Main.camera3D.getTransforms().set(1, new Rotate(Main.cameraRotAngleX, Rotate.X_AXIS));
        System.out.println(Main.camera3D.getTransforms().size());
        previousX=currentX;
        previousY=currentY;
        }
	@FXML
	public void on3DViewClick(MouseEvent event){
		previousX=event.getX();
		previousY=event.getY();
	}
	
	@FXML
	public void on3DViewScroll(ScrollEvent event){
		if(event.getDeltaY()>0){
			Main.translateZ+=10;
			if(Main.translateZ>0) Main.translateZ=0;
		} else{
			Main.translateZ-=10;
		}
		Main.camera3D.getTransforms().set(2, new Translate(0, 0, Main.translateZ));
		System.out.println("Scrolling! "+event.getDeltaY());
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
	private void updateSelectedLayerImage(int layerIndex){
		if(Main.selectedTomographyScan!=null){
			Main.currentLayerImage=Main.selectedTomographyScan.getLayerImage(layerIndex);
			BufferedImage bImage = ImageVisualization.convDenseMatrixToBufImage(Main.currentLayerImage);
			WritableImage image = new WritableImage(bImage.getWidth(), bImage.getHeight());
			SwingFXUtils.toFXImage(bImage, image);
			double threshold = 0.95;
			threshold = Double.parseDouble(textFieldSegmentationThreshold.getText());
			if(threshold<0||threshold>1) threshold = 0.95;
			
			if((cbShowSegmentation.isSelected())&&(Main.volumeVesselSegmentationData!=null)&&(layerIndex>=Main.segmentatedDataRange[0])&&(layerIndex<=Main.segmentatedDataRange[1])){
				paintInVessels(image, Main.volumeVesselSegmentationData.getLayer(layerIndex-Main.segmentatedDataRange[0]), threshold);
			}
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
	// method for painting int segmentated blood vessels in image
	private void paintInVessels(WritableImage image, DenseMatrix64F segmentationData, double threshold){
		PixelWriter pWriter = image.getPixelWriter();
		
		for(int i=0; i<image.getHeight(); i++){
			for(int j=0; j<image.getWidth(); j++){
				if(segmentationData.get(i,j)>=threshold){
					Color color = new Color(0,1,0,1);
					pWriter.setColor(j, i, color);
				}
			}
		}
	}
	
}
