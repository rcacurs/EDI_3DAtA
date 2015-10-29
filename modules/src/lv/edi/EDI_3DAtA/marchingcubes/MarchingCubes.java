package lv.edi.EDI_3DAtA.marchingcubes;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Vector;

import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import lv.edi.EDI_3DAtA.common.VolumetricData;

/** Class for marching cubes algorithm execution
 * 
 * @author Ricards Cacurs
 *
 */
public class MarchingCubes {
	private VolumetricData data; // reference to targe data on wich marching cubes is to be performed
	private DoubleProperty progress; // progress for isosurface extraction
	/**
	 * Constructor for MarhincCubes iso-surface generator
	 * @param data reference to VolumetricData object from which to extract iso-surface
	 */
	public MarchingCubes(VolumetricData data){
		this.progress=new SimpleDoubleProperty();
		this.data=data;
	}
	
//	/** Returns list of vertices for isosurface
//	 * 
//	 * @param isovalue - isovalue for wich surface is generated
//	 * @return ArrayList<DenseMatrix64F> of vertices for iso-surface
//	 */
//	public ArrayList<DenseMatrix64F> generateIsoSurface(double isovalue){
//		ArrayList<DenseMatrix64F> vertices = new ArrayList<DenseMatrix64F>(100);
//		ArrayList<DenseMatrix64F> currentVertexList;
//		DenseMatrix64F offset = new DenseMatrix64F(3,1);
//		CubeMC cube = new CubeMC();
//		if(data.size()!=0){
//			int maxX = data.getLayer(0).numCols;
//			int maxY = data.getLayer(0).numRows;
//			int maxZ = data.size();
//
//			for(int k=0; k<maxZ-1; k++){
//				
//				offset.set(2, 0, k);
//				for(int j=0; j<maxY-1; j++){
//					offset.set(1,0, j);
//					for(int i=0; i<maxX-1; i++){
//						offset.set(0,0,i);
//						cube.setVertexDensities(data.getValuesAtCubeVertices(i, j, k));
//						currentVertexList=cube.getVertexList(isovalue);
//						for(int ind=0; ind<currentVertexList.size(); ind++){
//							CommonOps.add(currentVertexList.get(ind), offset, currentVertexList.get(ind));
//						}
//						vertices.addAll(currentVertexList);
//						
//					}
//				}
//			}
//		}
//		
//		return vertices;
//	}
	/**
	 * Method generates iso-surface model. Data is returned as TriangleMeshData.
	 *  @param  isovalue - value at which to generate iso-surface
	 *	@return TriangleMeshData - generated iso-surface in TriangleMeshData format.
	 */
	public TriangleMeshData generateIsoSurface(double isovalue){
		ArrayList<Integer> facesal = new ArrayList<Integer>();
		ArrayList<Float> vertices = new ArrayList<Float>();
		ArrayList<DenseMatrix64F> currentVertexList;
		
		DenseMatrix64F offset = new DenseMatrix64F(3,1);
		CubeMC cube = new CubeMC();
		float xMax=Float.MIN_VALUE;;
		float yMax=Float.MIN_VALUE;;
		float zMax=Float.MIN_VALUE;
		float xMin=Float.MAX_VALUE;
		float yMin=Float.MAX_VALUE;
		float zMin=Float.MAX_VALUE;
		float tempx;
		float tempy;
		float tempz;
		if(data.size()!=0){
			int maxX = data.getLayer(0).numCols;
			int maxY = data.getLayer(0).numRows;
			int maxZ = data.size();
			int indexCounter=0;
			for(int k=0; k<maxZ-1; k++){
				progress.set((double)k/(maxZ-1));
				offset.set(2, 0, k);
				for(int j=0; j<maxY-1; j++){
					offset.set(1,0, j);
					for(int i=0; i<maxX-1; i++){
						offset.set(0,0,i);
						cube.setVertexDensities(data.getValuesAtCubeVertices(i, j, k));
						currentVertexList=cube.getVertexList(isovalue);
						for(int ind=0; ind<currentVertexList.size(); ind++){
							CommonOps.add(currentVertexList.get(ind), offset, currentVertexList.get(ind));
						}
						for(int z=0; z<currentVertexList.size(); z++){
							tempx=(float) currentVertexList.get(z).get(0);
							tempy=(float) currentVertexList.get(z).get(1);
							tempz=(float) currentVertexList.get(z).get(2);
							vertices.add(tempx);
							vertices.add(tempy);
							vertices.add(tempz);
							facesal.add(indexCounter);
							facesal.add(0); // add texture index as zero because texture is not used
							indexCounter++;
							
							if(tempx>xMax){
								xMax=tempx;
							}
							if(tempx<xMin){
								xMin=tempx;
							}
							if(tempy>yMax){
								yMax=tempy;
							}
							if(tempy<yMin){
								yMin=tempy;
							}
							if(tempz>zMax){
								zMax=tempz;
							}
							if(tempz<zMin){
								zMin=tempz;
							}
						}
						
					}
				}
			}
		}
		progress.set(1);
		float[] points = new float[vertices.size()];
		for(int i=0; i<vertices.size(); i++){
			points[i]=vertices.get(i);
		}
		int[] faces = new int[facesal.size()];
		for(int i=0; i<facesal.size(); i++){
			faces[i]=facesal.get(i);
		}
		float[] texCoords = new float[2];
		float[] center = new float[3];
		center[0]=(xMax+xMin)/2;
		center[1]=(yMax+yMin)/2;
		center[2]=(zMax+zMin)/2;
		return new TriangleMeshData(points, texCoords, faces, center);
			
	}
	/** method that reduces number of vertexes by removing vertexes with same coordinates
	 *  this method changes input ArrayList of points. The index list for triangulation 
	 *  of these points is returned by this function. Indexing is started from 1.
	 *  @param inputVertexes array list of calculated vertexes
	 *  @return index list for triangle faces
	 */
	public static ArrayList<Integer> removeDuplicatePoints(ArrayList<DenseMatrix64F> inputVertexes){
		ArrayList<Integer> indexes = new ArrayList<Integer>(inputVertexes.size());
		ArrayList<DenseMatrix64F> reducedVertexes = new ArrayList<DenseMatrix64F>(inputVertexes.size()/10);
		reducedVertexes.add(inputVertexes.get(0));
		indexes.add(1);
		int lastOriginalIndex=1;
		boolean isOriginal=true;
		for(int i=1; i<inputVertexes.size(); i++){
			isOriginal=true;
			System.out.println("Prcessing input vertex: "+i);
			for(int j=0; j<reducedVertexes.size(); j++){
				if((reducedVertexes.get(j).data[0]==inputVertexes.get(i).data[0])&&
					(reducedVertexes.get(j).data[1]==inputVertexes.get(i).data[1])&&
					(reducedVertexes.get(j).data[2]==inputVertexes.get(i).data[2])){
					isOriginal=false;
					indexes.add(j+1);
				} 
			}
			if(isOriginal){
				lastOriginalIndex++;
				indexes.add(lastOriginalIndex);
				reducedVertexes.add(inputVertexes.get(i));
			}
		}
		inputVertexes = reducedVertexes;
		return indexes;
	}
	
	
	/** saves ArrayList of vertices in .obj file
	 * 
	 * @param vertices vertices in ArrayList{@literal <}DenseMatrix64F{@literal >} format
	 * @param filename filename of destination file.
	 * @throws FileNotFoundException throws if file not found or cannot be created!
 	 */
	public static void saveVerticesToObj(ArrayList<DenseMatrix64F> vertices, String filename) throws IOException{
		//ArrayList<Integer> faceIndexes = removeDuplicatePoints(vertices);;
		
		
		Vector<String> stringVector = new Vector<String>();
		long tick1 = System.currentTimeMillis();
		for(DenseMatrix64F item:vertices){
			String vertexS = String.format(Locale.ENGLISH, "v %5.2f %5.2f %5.2f", item.get(0), item.get(1), item.get(2));
			stringVector.add(vertexS);
		}

		for(int i=0; i<vertices.size()/3; i++){
			String indexS = "f " +(i*3+1)+" "+(i*3+2)+" "+(i*3+3);
			stringVector.add(indexS);
		}
		
		long tick2 = System.currentTimeMillis();
		System.out.printf("Time for forming string vector: %d [ms]\n", tick2-tick1);
		
		File textFile = new File(filename);
		if(textFile.exists()){
			textFile.delete();
		}
		
		RandomAccessFile RAMFilet = new RandomAccessFile(textFile, "rw");
		FileChannel rwChannel = RAMFilet.getChannel();
		
		tick1 = System.currentTimeMillis();
		byte[] buffer = String.join("", stringVector).getBytes();
		tick2 = System.currentTimeMillis();
		System.out.printf("Time for joining strings %d\n",tick2-tick1);
		tick1=System.currentTimeMillis();
		ByteBuffer wrBuf = rwChannel.map(FileChannel.MapMode.READ_WRITE, 0, buffer.length);
		
		wrBuf.put(buffer);	
		rwChannel.close();
		buffer = null;
		tick2=System.currentTimeMillis();
		System.out.printf("Time for filling file: %d [ms]\n");
	}
	/**
	 * Getter method for progress property
	 * @return DoubleProperty progress of isovalue generation propery
	 */
	public DoubleProperty getProgressProperty(){
		return progress;
	}
}
