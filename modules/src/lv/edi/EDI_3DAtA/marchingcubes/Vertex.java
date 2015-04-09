package lv.edi.EDI_3DAtA.marchingcubes;

/**
 * Class representing vertex (3 dimmensional point)
 * @author Ricards Cacurs
 *
 */
public class Vertex {
	private double[] data = new double[3]; // initialize vertex data
	
	/**
	 * Vertex constructor
	 * @param data - data array of whose length is 3, representing one vertex
	 */
	public Vertex(double[] data){
		this.data = data;
	}
	
	/**
	 * Constructor for vertex
	 * @param x - x coordinate
	 * @param y - y coordinate
	 * @param z - z coordinate
	 */
	public Vertex(double x, double y, double z){
		this.data[0] = x;
		this.data[1] = y;
		this.data[2] = z;
	}
	
	/**
	 * return x coordinate of vertex
	 * @return double x
	 */
	public double getX(){
		return data[0];
	}
	
	/**
	 * returns y coordinate of vertex
	 * @return double z
	 */
	public double getY(){
		return data[1];
	}
	
	/**
	 * returns z coordinate of vertex
	 * @return double z
	 */
	
	public double getZ(){
		return data[2];
	}
	
	/** tanslates vertex by vector represented by vertex
	 * 
	 * @param vertex where to translate this vertex
	 */
	public void translate(Vertex vertex){
		this.data[0]=this.data[0]+vertex.getX();
		this.data[1]=this.data[1]+vertex.getY();
		this.data[2]=this.data[2]+vertex.getZ();
	}
	
	
}
