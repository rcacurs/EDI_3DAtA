package lv.edi.EDI_3DAtA.marchingcubes;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

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
	
	/**
	 * Constructor for MarhincCubes iso-surface generator
	 * @param data reference to VolumetricData object from which to extract iso-surface
	 */
	public MarchingCubes(VolumetricData data){
		this.data=data;
	}
	
	/** Returns list of vertices for isosurface
	 * 
	 * @param isovalue - isovalue for wich surface is generated
	 * @return ArrayList<DenseMatrix64F> of vertices for iso-surface
	 */
	public ArrayList<DenseMatrix64F> generateIsoSurface(double isovalue){
		ArrayList<DenseMatrix64F> vertices = new ArrayList<DenseMatrix64F>(100);
		ArrayList<DenseMatrix64F> currentVertexList;
		DenseMatrix64F offset = new DenseMatrix64F(3,1);
		CubeMC cube = new CubeMC();
		if(data.size()!=0){
			int maxX = data.getLayer(0).numCols;
			int maxY = data.getLayer(0).numRows;
			int maxZ = data.size();
			
			for(int k=0; k<maxZ-1; k++){
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
						vertices.addAll(currentVertexList);
						
					}
				}
			}
		}
		return vertices;
	}
	
	
	/** saves ArrayList of vertices in .obj file
	 * 
	 * @param vertices vertices in ArrayList<DenseMatrix64F> format
	 * @throws FileNotFoundException throws if file not found or cannot be created!
 	 */
	public static void saveVerticesToObj(ArrayList<DenseMatrix64F> vertices, String filename) throws FileNotFoundException{
		PrintWriter writer = new PrintWriter(filename);
		for(DenseMatrix64F item:vertices){
			String vertexS = "v "+item.get(0)+" "+item.get(1)+" "+item.get(2);
			writer.println(vertexS);
		}
		for(int i=0; i<vertices.size()/3; i++){
			String indexS = "f " +(i*3+1)+" "+(i*3+2)+" "+(i*3+3);
			writer.println(indexS);
		}
		writer.close();
	}
}
