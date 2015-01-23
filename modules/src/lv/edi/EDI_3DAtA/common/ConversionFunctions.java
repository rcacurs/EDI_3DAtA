package lv.edi.EDI_3DAtA.common;

import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageFloat32;

/**
 * 
 * @author Ričards Cacurs
 * Class for simple conversions
 *
 */
public class ConversionFunctions {
	
	/** Function returns string representation of image in
	 * 
	 * @param inputImage
	 * @return
	 */
	public static String imageToString(ImageBase inputImage){
		
		StringBuilder sb = new StringBuilder();
	switch(inputImage.getImageType().getDataType()){
	case F32:
		
		for(int i=0; i<inputImage.height; i++){
			for(int j=0; j<inputImage.width; j++){
				sb.append(((ImageFloat32)inputImage).get(j, i));
					if(j<(inputImage.width-1)){
						sb.append(",");
					}
				}
				sb.append("\n");
			}
		break;
	default: 
		break;
	}
	return sb.toString();	
	}
}
