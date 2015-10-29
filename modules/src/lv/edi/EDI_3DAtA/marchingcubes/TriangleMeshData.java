package lv.edi.EDI_3DAtA.marchingcubes;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Locale;
import java.util.Vector;

import org.ejml.data.DenseMatrix64F;

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
		
		ByteBuffer bbuffer = ByteBuffer.allocateDirect((vertices.length/3+faces.length/6)*25);
		long tick1 = System.currentTimeMillis();
		int numBytes = 0;
		for(int i=0; i<vertices.length/3; i++){
			String vertexS = String.format(Locale.ENGLISH, "v %5.2f %5.2f %5.2f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
			byte[] buf = vertexS.getBytes();
			bbuffer.put(buf);
			numBytes+=buf.length;
			buf=null;
			vertexS=null;
		}

		for(int i=0; i<faces.length/6; i++){
			String indexS = "f " +(faces[i*6]+1)+" "+(faces[i*6+2]+1)+" "+(faces[i*6+4]+1)+"\n";
			byte[] buf = indexS.getBytes();
			bbuffer.put(buf);
			numBytes+=buf.length;
			buf=null;
			indexS=null;
		}
		
		long tick2 = System.currentTimeMillis();
		System.out.printf("Time for forming byte buffer: %d [ms]\n", tick2-tick1);
		
		File textFile = new File(fileName);
		if(textFile.exists()){
			textFile.delete();
		}
		
		RandomAccessFile RAMFile = new RandomAccessFile(textFile, "rw");
		FileChannel rwChannel = RAMFile.getChannel();
		

		
		System.out.printf("Time for joining strings %d\n",tick2-tick1);
		tick1=System.currentTimeMillis();
		ByteBuffer wrBuf = rwChannel.map(FileChannel.MapMode.READ_WRITE, 0, numBytes);
		bbuffer.rewind();
		while (wrBuf.hasRemaining())
	         wrBuf.put(bbuffer.get()); 
		bbuffer.clear();
		bbuffer = null;
		rwChannel.close();
		RAMFile.close();
		
		tick2=System.currentTimeMillis();
		System.out.printf("Time for filling file: %d [ms]\n", tick2-tick1);
		
	}
}
