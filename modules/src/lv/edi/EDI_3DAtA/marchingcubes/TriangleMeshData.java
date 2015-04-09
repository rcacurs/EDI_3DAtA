package lv.edi.EDI_3DAtA.marchingcubes;

import java.io.IOException;
import java.io.PrintWriter;

/**
 * 
 * @author Ricards Cacurs
 * Class that packs triangular mesh dasta.
 */


public class TriangleMeshData {
	public final float[] vertices;
	public final float[] texCoords;
	public final int[] faces;
	public final float[] center;
	
	
	/**
	 * Construct mesh data object
	 * @param vertices array of vertices
	 * @param texCoords	array of texture coordinates
	 * @param faces array of faces
	 * @param center coordinates for center point of model
	 */
	public TriangleMeshData(float[] vertices, float[] texCoords, int[] faces, float[] center){
		this.vertices = vertices;
		this.faces = faces;
		this.texCoords = texCoords;
		this.center = center;
	}
	
	/**
	 * Function allow to save mesh in .obj file.
	 * @param fileName filename of object file. File name could contain path to object file
	 * @throws IOException if file cannot be created. 
	 */
	public void saveAsObj(String fileName) throws IOException{
		PrintWriter writer = new PrintWriter(fileName);
		
		for(int i=0; i<vertices.length/3; i++){
			String vertexS = String.format("v %5.2f %5.2f %5.2f", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
			writer.println(vertexS);
		}
		
		for(int i=0; i<faces.length/6; i++){
			String indexS = "f " +(faces[i*6]+1)+" "+(faces[i*6+2]+1)+" "+(faces[i*6+4]+1);
			writer.println(indexS);
		}
		writer.close();
		
	}
}
