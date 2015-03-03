package lv.edi.EDI_3DAtA.marchingcubes;
/**
 * 
 * @author Ričards Cacurs
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
	 */
	public TriangleMeshData(float[] vertices, float[] texCoords, int[] faces, float[] center){
		this.vertices = vertices;
		this.faces = faces;
		this.texCoords = texCoords;
		this.center = center;
	}
}
