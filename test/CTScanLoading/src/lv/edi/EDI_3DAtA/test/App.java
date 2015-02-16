package lv.edi.EDI_3DAtA.test;

import java.io.File;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;

public class App extends Application{
	File scanFile;
	@Override
	public void start(Stage stage) throws Exception {
		
		Button button = new Button();
		button.setText("Select CT Scan file");
		button.setOnAction(new EventHandler<ActionEvent>(){
			@Override
			public void handle(ActionEvent arg0) {
				FileChooser fchooser = new FileChooser();
				fchooser.setTitle("Locate Tomography Scan File");
				fchooser.getExtensionFilters().add(new ExtensionFilter("CT Scan files", "*.mhd"));
				scanFile = fchooser.showOpenDialog(stage);
				System.out.println("File selected: "+scanFile);
			}
		});
		
		StackPane root = new StackPane();
		root.getChildren().add(button);
		
		Scene scene = new Scene(root, 300, 250);
		
		stage.setTitle("Hello World!");
		stage.setScene(scene);
		stage.show();
		
	}
	
	public static void main(String[] args){
		launch(args);
	}

}
