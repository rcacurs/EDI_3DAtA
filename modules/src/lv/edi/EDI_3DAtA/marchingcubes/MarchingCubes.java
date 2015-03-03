package lv.edi.EDI_3DAtA.marchingcubes;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import lv.edi.EDI_3DAtA.common.VolumetricData;

/** Class for marching cubes algorithm execution
 * 
 * @author Ričards Cacurs
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
		float xSum=0;
		float ySum=0;
		float zSum=0;
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
							vertices.add((float) currentVertexList.get(z).get(0));
							xSum=xSum+(float) currentVertexList.get(z).get(0);
							vertices.add((float) currentVertexList.get(z).get(1));
							ySum=ySum+(float) currentVertexList.get(z).get(1);
							vertices.add((float) currentVertexList.get(z).get(2));
							zSum=zSum+(float) currentVertexList.get(z).get(2);
							facesal.add(indexCounter);
							facesal.add(0); // add texture index as zero because tecture is not used
							indexCounter++;
						}
						
					}
				}
			}
		}
		progress.set(1);
		System.out.println("Vertices size: "+vertices.size());
		System.out.println("Faces size: "+facesal.size());
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
		center[0]=xSum/vertices.size();
		center[1]=ySum/vertices.size();
		center[2]=zSum/vertices.size();
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
	 * @param vertices vertices in ArrayList<DenseMatrix64F> format
	 * @throws FileNotFoundException throws if file not found or cannot be created!
 	 */
	public static void saveVerticesToObj(ArrayList<DenseMatrix64F> vertices, String filename) throws FileNotFoundException{
		//ArrayList<Integer> faceIndexes = removeDuplicatePoints(vertices);;
		
		PrintWriter writer = new PrintWriter(filename);
		for(DenseMatrix64F item:vertices){
			String vertexS = String.format("v %5.2f %5.2f %5.2f", item.get(0), item.get(1), item.get(2));
			writer.println(vertexS);
		}
		for(int i=0; i<vertices.size()/3; i++){
			String indexS = "f " +(i*3+1)+" "+(i*3+2)+" "+(i*3+3);
			writer.println(indexS);
		}
		writer.close();
	}
	/**
	 * Getter method for progress property
	 * @return DoubleProperty progress of isovalue generation propery
	 */
	public DoubleProperty getProgressProperty(){
		return progress;
	}
}
