package lv.edi.EDI_3DAtA.opencvcudainterface;

public class Compute {
	
	public native double[] gaussianBlur(double[] input, int rows, int cols);
	public native double[] segmentBloodVessels(double[] input, int rows, int cols, double[] codes, double[] means, int patchSize, int numberOfFilters, double[] model, double[] scaleparamsMeans, double[] scaleparamsSd, double[] imageMask);
	
	public Compute(){
		System.loadLibrary("bloodVesselSegmentation");
	}

}
