package lv.edi.EDI_3DAtA.vessel2objapp;

import java.net.URL;
import java.util.ResourceBundle;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Hyperlink;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class MiscController implements Initializable{
	@FXML
	private Hyperlink hyperlinkEdi;
	@FXML
	private ImageView imageViewLogo;
	@Override
	public void initialize(URL location, ResourceBundle resources) {
		// TODO Auto-generated method stub
		Image ediLogo = new Image(getClass().getResourceAsStream("edilogo.png"));
		imageViewLogo.setImage(ediLogo);
	}
	
	@FXML
	public void onHyperlinkEdi(ActionEvent event){
		Main.hostServices.showDocument("http://www.edi.lv");
		hyperlinkEdi.setVisited(false);
	}

}
