package lv.edi.EDI_3DAtA.marchingcubes;

import java.util.ArrayList;

import org.ejml.data.DenseMatrix64F;

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
		if(data.size()!=0){
			int maxX = data.getLayer(0).numCols;
			int maxY = data.getLayer(1).numRows;
			int maxZ = data.size();
			
			for(int z=0; z<maxX-1; z++){
				for(int y=0; y<maxY-1; y++){
					for(int x=0; z<maxZ-1; z++){
						
					}
				}
			}
		}
		
		return vertices;
		
	}
}
