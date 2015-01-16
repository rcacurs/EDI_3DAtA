package lv.edi.EDI_3DAtA.imageio;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import lv.edi.EDI_3DAtA.common.VolumetricData;

import org.ejml.data.DenseMatrix64F;

import boofcv.struct.image.ImageUInt8;


/**
 * @author Ričards Cacurs
 * Class represents data in medical image format - meta image (.mhd .raw)
 * 
 */
public class MetaImage {
	private String objectType;
	private int nDims;
	private boolean isBinaryData;
	private boolean isBinaryDataByteOrderMSB=false;
	private boolean isCompressedData;
	private DenseMatrix64F transformMatrix;
	private DenseMatrix64F offset;
	private DenseMatrix64F centerOfRotation;
	private String anatomicalOrientation;
	private DenseMatrix64F elementSpacing;
	private DenseMatrix64F dimSize;
	private MetaElementType elementType;
	private File elementDataFile;
	private File elementHeaderFile;
	
	/**
	 * Constructs MetaImage object.
	 * 
	 * @param metaHeaderFile String that specifies path to the meta image header file (Can relative path or absolute path)
	 * @throws FileNotFoundException if file specified in constructor argument doesn't exist.
	 * @throws IOException if exception occurs while reading specified file.
	 */
	public MetaImage(String metaHeaderFile) throws FileNotFoundException, IOException{
		elementHeaderFile = new File(metaHeaderFile);
		
		BufferedReader br = new BufferedReader(new FileReader(elementHeaderFile));
		String line;
		String[] lineElements;
		
		while((line = br.readLine()) != null){
			lineElements = line.split(" ");
			
			switch(lineElements[0]){
			case "ObjectType":
				setObjectType(lineElements[2]);
				break;
			case "NDims":
				setNDims(lineElements[2]);
				break;
			case "BinaryData":
				setIsBinaryData(lineElements[2]);
				break;
			case "BinaryDataByteOrderMSB":
				setIsbinaryDataByteOrderMSB(lineElements[2]);
				break;
			case "CompressedData":
				setIsCompressedData(lineElements[2]);
				break;
			case "TransformMatrix":
				String trMatrix="";
				for(int i=2; i<lineElements.length; i++){
					trMatrix+=lineElements[i]+" ";
				}
				setTransformMatrix(trMatrix);
				break;
			case "Offset":
				String offs="";
				for(int i=2; i<lineElements.length; i++){
					offs+=lineElements[i]+" ";
				}
				setOffset(offs);
				break;
			case "CenterOfRotation":
				String cor="";
				for(int i=2; i<lineElements.length; i++){
					cor+=lineElements[i]+" ";
				}
				setCenterOfRotation(cor);
				break;
			case "AnatomicalOrientation":
				setAnatomicalOrientation(lineElements[2]);
				break;
			case "ElementSpacing":
				String elemSpacing="";
				for(int i=2; i<lineElements.length; i++){
					elemSpacing+=lineElements[i]+" ";
				}
				setElementSpacing(elemSpacing);
				break;
			case "DimSize":
				String dimSz="";
				for(int i=2; i<lineElements.length; i++){
					dimSz+=lineElements[i]+" ";
				}
				setDimSize(dimSz);
				break;
			case "ElementType":
				setElementType(lineElements[2]);
				break;
			case "ElementDataFile":
				setElementDataFile(lineElements[2]);
				break;
			default:
				break;
			}
		}
		br.close();
		
	}
	/**
	 * Method for setting meta image objectType property from string.
	 * 
	 * @param objectType value for objectType property
	 */
	public void setObjectType(String objectType){
		this.objectType=objectType;
	}
	
	/**
	 * Method for setting meta image nDimensions from string.
	 * 
	 * @param nDims parameter value represented as string. Specifies number of dimensions of Meta Image file.
	 */
	public void setNDims(String nDims){
		this.nDims = Integer.parseInt(nDims);
	}
	/**
	 * Method for setting isBinaryData flag.
	 * 
	 * @param isBinaryData parameter value as string ignoring case.
	 */
	public void setIsBinaryData(String isBinaryData){
		this.isBinaryData = Boolean.parseBoolean(isBinaryData);
	}
	
	/** Method for setting Endianness of data.
	 * 
	 * @param isBinaryDataByteOrderMSB string boolean value representing endianness ignoring case.
	 * if MSB endianness is used it should be set to true.
	 * 
	 */
	public void setIsbinaryDataByteOrderMSB(String isBinaryDataByteOrderMSB){
		this.isBinaryDataByteOrderMSB = Boolean.parseBoolean(isBinaryDataByteOrderMSB);
	}
	
	/**
	 * Method sets if data is compressed.
	 * 
	 * @param isCompressedData string arguments representing boolean value if data is compressed. Case is ignored.
	 * example "true" for true value. "false" for false value.
	 */
	public void setIsCompressedData(String isCompressedData){
		this.isCompressedData = Boolean.parseBoolean(isCompressedData);
	}
	
	/**
	 * Method sets transformation matrix.
	 * @param transformMatrix string containing transformation matrix values separated with space in Row-Major order.
	 *  example: setTransformMatrix("1 0 0 0 1 0 0 0 1"). In case of incorrect format transformMatrix is set to null
	 */
	public void setTransformMatrix(String transformMatrix){
		String[] elements = transformMatrix.split(" ");
		if(elements.length == 9){
			DenseMatrix64F mat = new DenseMatrix64F(3, 3);
			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					mat.set(i, j, Double.parseDouble(elements[i*3+j]));
				}
			}
			this.transformMatrix = mat;
		} else{
			this.transformMatrix = null;
		}	
	}
	
	/**
	 * Method sets offset parameter for meta image
	 * 
	 * @param offset parameter value in string format. example: setOffset("1 1 1"). Must be three values separated with space.
	 */
	public void setOffset(String offset){
		String[] elements = offset.split(" ");
		
		if(elements.length == 3){
			DenseMatrix64F mat = new DenseMatrix64F(3,1);
			for(int i=0; i<3; i++){
				mat.set(i, 0, Double.parseDouble(elements[i]));
			}
			this.offset = mat;
		} else{
			this.offset = null;
		}
		
	}
	
	/**
	 * Methods for setting meta image centerOfRotation property.
	 * 
	 * @param centerOfRotation center of rotation property value
	 */
	public void setCenterOfRotation(String centerOfRotation){
		String[] elements = centerOfRotation.split(" ");
		
		if(elements.length == 3){
			DenseMatrix64F mat = new DenseMatrix64F(3,1);
			for(int i=0; i<3; i++){
				mat.set(i, 0, Double.parseDouble(elements[i]));
			}
			this.centerOfRotation = mat;
		} else{
			this.centerOfRotation = null;
		}
	}
	
	/**
	 * Method set anatomicalOrientation property of meta image file.
	 * @param anatomicalOrientation string representing anatomical orientation property
	 */
	public void setAnatomicalOrientation(String anatomicalOrientation){
		this.anatomicalOrientation = anatomicalOrientation;
	}
	
	/** Method set anatomicalSpacing property for meta image.
	 * 
	 * @param elementSpacing string representing physical distance between elements(voxels). Distance between elements
	 * is specified in three directions - x, y, z in [mm] and separated with space: setElementSpacing("0.7 0.7 1"); 
	 */
	public void setElementSpacing(String elementSpacing){
		String[] elements = elementSpacing.split(" ");
		
		if(elements.length == 3){
			DenseMatrix64F mat = new DenseMatrix64F(3, 1);
			for(int i=0; i<3; i++){
				mat.set(i,0, Double.parseDouble(elements[i]));
			}
			this.elementSpacing = mat;
		}else {
			this.elementSpacing = null;
		}
	}
	
	/**
	 * Method setting dimension size for meta image file representing scan. Specifies resolution.
	 * @param dimSize string representing dimension size(resolution) in x, y and z directions of the scan. example: setDimSize("512 512 300");
	 */
	public void setDimSize(String dimSize){
		String[] elements = dimSize.split(" ");
		
		if(elements.length == 3){
			DenseMatrix64F mat = new DenseMatrix64F(3,1);
			for(int i=0; i<3; i++){
				mat.set(i, 0, Double.parseDouble(elements[i]));
			}
		    this.dimSize = mat;
		} else{
			this.dimSize = null;
		}
	}
	
	/** Method sets data of the scan stored in .raw file
	 * 
	 * @param elementType MetaElementType enum representing element types.
	 *  currently supported META_SHORT
	 */
	public void setElementType(String elementType){
		switch(elementType){
		case "MET_SHORT":
			this.elementType=MetaElementType.MET_SHORT;
			break;
		default:
			this.elementType = null;
			break;		
		}
	}
	
	/**
	 *  Method sets element header file object.
	 *  
	 *  @param headerFile String representing path name to file. relative or absolute.
	 */
	public void setElementHeaderFile(String headerFile){
		this.elementHeaderFile = new File(headerFile); 
	};
	
	/**
	 * Sets File object representing data file .raw.
	 * 
	 * @param dataFile String representing data File. 
	 *  Can be specified like single file name: "example.raw"
	 *   or path relative to header file header.mhd location)
	 */
	public void setElementDataFile(String dataFile){
		
		if(elementHeaderFile != null){
			String elementHeaderFileDirectoryStr = elementHeaderFile.getParent();
			this.elementDataFile = new File(elementHeaderFileDirectoryStr+"/"+dataFile);
		}else{
			dataFile = null;
		}
		
	}
	
	/**
	 * Method for getting objectType property of meta image.
	 * @return objectType property of mete image.
	 */
	public String getObjectType(){
		return objectType;
	}
	
	/**
	 * Method for getting nDims (number of dimensions) property of meta image.
	 * @return returns integer representing dimensions of meta image file data.
	 */
	public int getNDims(){
		return nDims;
	}
	
	/**
	 * Method returns if meta image data file contains binary data
	 * @return return true if data file contains binary data.
	 */
	public boolean isBinaryData(){
		return isBinaryData;
	}
	
	/**
	 * Methods returns if saved data is in MSB byte ordering
	 * @return returns true if byte ordering is MSB, false if LSB.
	 */
	public boolean isBinaryDataByteOrderMSB(){
		return isBinaryDataByteOrderMSB;
	}
	
	/**
	 * Method returns if data is compressed.
	 * @return returns true if data is compressed, false if not compressed
	 */
	public boolean isCompressedData(){
		return isCompressedData;
	}
	
	/**
	 * Method for getting transform matrix of meta image
	 * 
	 * @return returns 3x3 DenseMatrix64F transformation matrix (Meta image property TransformMatrix)
	 */
	public DenseMatrix64F getTransformMatrix(){
		return transformMatrix;
	}
	
	/**
	 * Methods for getting offset matrix of meta image
	 * 
	 * @return matrix containing meta image offset property.
	 */
	public DenseMatrix64F getOffset(){
		return offset;
	}
	
	/**
	 * Methods for getting centerOfRotation property of meta image
	 * 
	 * @return centerOfRotation property of meta image.
	 */
	public DenseMatrix64F getCenterOfRotation(){
		return centerOfRotation;
	}
	
	/**
	 * Method get anatomicalOrientation property of meta image.
	 * @return string representing anatomical orientation property of meta image.
	 */
	public String getAnatomicalOrientation(){
		return anatomicalOrientation;
	}
	
	/**
	 * Method getting elementSpacing property of meta image. 
	 * elementSpacing property shows physical spacing between voxels of the scan
	 * 
	 * @return DenseMatrix64F 3x1 matrix/vector representing spacing in [x, y, z] dimensions for the scan
	 */
	
	public DenseMatrix64F getElementSpacing(){
		return elementSpacing;
	}
	
	/**
	 * Method for getting dimSize property of meta image. Basically return resolution of the scan.
	 * 
	 * @return DenseMatrix54F 3x1 matrix specifying dimensions in each direction x, y, z.
	 */
	public DenseMatrix64F getDimSize(){
		return dimSize;
	}
	
	/**
	 * Method for getting elementType property of meta image.
	 * 
	 * @return MetaElementType enum representing the type of data in .raw file.
	 */
	public MetaElementType getElementType(){
		return elementType;
	}
	
	/**
	 * Method of getting element header file(.mhd) object
	 * 
	 * @return File object representing elementHeaderFile
	 */
	public File getElementHeaderFile(){
		return elementHeaderFile;
	}
	
	/**
	 * Method for getting element data file (.raw) object
	 * 
	 * @return File object representing elementDataFile.
	 */
	public File getElementDataFile(){
		return elementDataFile;
	}
	
	/**
	 * String representation of object.
	 * 
	 * @return String return string representation of MetaImage object.
	 */
	public String toString(){
		String str="";
		str+="Header File Path: "+getElementHeaderFile()+"\n\n";
		str+="Data File Path: "+getElementDataFile()+"\n\n";
		str+="ObjectType: "+getObjectType()+"\n\n";
		str+="BinaryData: "+isBinaryData()+"\n\n";
		str+="BinaryDataByteOrderMSB: "+isBinaryDataByteOrderMSB()+"\n\n";
		str+="CompressedData: "+isCompressedData()+"\n\n";
		str+="TransformMatrix: "+getTransformMatrix()+"\n";
		str+="Offset: "+getOffset()+"\n";
		str+="CenterOfOrientation: "+getCenterOfRotation()+"\n";
		str+="AnatomicalOrientation: "+getAnatomicalOrientation()+"\n\n";
		str+="ElementSpacing: "+getElementSpacing()+"\n";
		str+="DimSize: "+getDimSize()+"\n";
		str+="ElementType: "+getElementType()+"\n\n";
		return str;
	}
	/**
	 * Function for getting volumetric data from Meta Image file. Currently supports only meta images that are stored
	 * And currently assumes RAS anatomical orientation.
	 * in META_SHORT type
	 * @return VolumetricData object acquired from meta image file. Returns null if data cannot be acquired.
	 */
	public VolumetricData getVolumetricData(){
		FileInputStream in;
		int readByteL;   // low address byte
		int readByteH;   // high address byte
		VolumetricData volData = new VolumetricData();
		if((elementDataFile!=null)&&(dimSize!=null)){
			try{
				in = new FileInputStream(elementDataFile);
			} catch(FileNotFoundException ex){
				return null;
			}
	
			for(int i=0; i<dimSize.get(2,0); i++){ // z axis
				int max_value=0;
				int min_value=Integer.MAX_VALUE;
				int[][] data = new int[(int)dimSize.get(1,0)][(int)dimSize.get(2,0)];
				for(int j=0; j<data[0].length; j++){ // y axis
					for(int k=0; k<data.length; k++){           // x axis
						try{
							readByteL = in.read() & 0xff;
							readByteH = in.read() & 0xff;
							if(!isBinaryDataByteOrderMSB()){
								data[k][j]=(int) (readByteL | (readByteH*256));
									
							} else{
								data[k][j]=(int)(readByteH | (readByteL*256));
							}
							if(data[k][j]>max_value){
								max_value=data[k][j];
							}
							if(data[k][j]<min_value){
								min_value=data[k][j];
							}
						}catch(IOException ex){
							try {
								in.close();
							} catch (IOException e) {
								return null;
							}
						}
					}
				}
				ImageUInt8 image = new ImageUInt8(data.length, data[0].length);
				for(int j=0; j<data[0].length; j++){
					for(int k=0; k<data.length; k++){
						image.set(k, j, (int)(255*(((double)(data[k][j]-min_value))/(max_value-min_value))));
					}
				}
				volData.addLayer(image);
			}
			try {
				in.close();
			} catch (IOException e) {

			}
			return volData;
		} else{
			return null;
		}
	}
	
}
