package lv.edi.EDI_3DAtA.opencvcudainterface;

public class Compute {
	
	public native double[] gaussianBlur(double[] input, int rows, int cols);
	
	
	public Compute(){
		System.loadLibrary("bloodVesselSegmentation");
	}

}
