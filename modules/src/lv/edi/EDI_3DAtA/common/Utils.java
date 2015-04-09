package lv.edi.EDI_3DAtA.common;
import org.ejml.data.DenseMatrix64F;
/**
 * 
 * @author Ricards Cacurs
 * 
 * This class is created for various utility functions for data formatting etc
 * 
 */
public class Utils {
	
	public static String denseMatrix64FtoCSVString(DenseMatrix64F inputMatrix){
		
		StringBuilder sb = new StringBuilder();
		
		for(int i=0; i<inputMatrix.numRows; i++){
			for(int j=0; j<inputMatrix.numCols-1; j++){
				sb.append(inputMatrix.get(i,j)+",");
			}
			sb.append(inputMatrix.get(i, inputMatrix.numCols-1)+"\n");
		}
		return sb.toString();
	}
}
