package lv.edi.EDI_3DAtA.bloodvesselsegm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import lv.edi.EDI_3DAtA.common.VolumetricData;

///**
// * 
// * @author Code - Ričards Cacurs.
// * 
// *  Class representing Stacked Multiscale Feature extractor from paper:
// *  Adam Coates and Andrew Y. Ng "Learning Feature Representations with K-means", 
// *  Neural Networks: Tricks of the Trade, 2nd edn, Springer, 2012.
// *
// */
public class SMFeatureExtractor {
//	private ArrayList<ImageFloat32> filterBank;
//	private VolumetricData volData;
//
//	/**
//	 * Constructor constructing feature extractor. 
//	 * @param filterBankFileName file name for filterbanks.csv file. It is CSV
//	 * file where each filter values are stored in separate lines,
//	 * and values are stored in row-major order.
//	 * @param kernelRows filter bank kernel row size
//	 * @param kernelCols filter bank kernel column size
//	 * @throws IOException throws exception if specified filename cannot be read
//	 */
//	public SMFeatureExtractor(VolumetricData volData, String filterBankFileName, int kernelRows, int kernelCols) throws IOException{
//		setFilterBank(filterBankFileName, kernelRows, kernelCols);
//		setVolData(volData);
//	}
//	
//	/**
//	 * Function return filter bank set in current Feature Extractor
//	 * @return the filterBank. Return null if filter bank not specified.
//	 */
//	public ArrayList<ImageFloat32> getFilterBank() {
//		return filterBank;
//	}
//
//	/**
//	 * Function for setting feature extractor filter-bank from .csv file containing
//	 * filter kernel values.
//	 * @param filterBankFileName path to the file. One row of file represents one filter values
//	 * Values for each kernel must be specified in Col-Maj order.
//	 * @param kernelRows number of filter kernel rows
//	 * @param kernelCols number of filter kernel columns
//	 */
//	public void setFilterBank(String filterBankFileName, int kernelRows, int kernelCols)
//	throws IOException{
//		
//		File filterBankFile = new File(filterBankFileName);
//		filterBank = new ArrayList<ImageFloat32>();
//		BufferedReader br = new BufferedReader(new FileReader(filterBankFile));
//		
//		String line;
//		String[] lineElements;
//		while((line=br.readLine())!=null){
//			ImageFloat32 filter = new ImageFloat32(kernelCols, kernelRows);
//			lineElements = line.split(",");
//			if(lineElements.length==(kernelRows*kernelCols)){
//				for(int i=0; i<kernelCols; i++){
//					for(int j=0 ;j<kernelRows; j++){
//						filter.set(i, j, Float.parseFloat(lineElements[kernelRows*i+j]));
//					}
//				}
//				filterBank.add(filter);
//			} else{
//				br.close();
//				throw new IOException("Specified format for kernel rows doesnt correpsond to contents of .csv file");
//			}
//		}
//		try{
//			br.close();
//		} catch(IOException ex){
//			
//		}
//	}
//	
//	/**
//	 * Function for accessing VolumetricData object for which FeatureExtractor currently is set.
//	 * @return 
//	 */
//	public VolumetricData getVolData() {
//		return volData;
//	}
//	
//	/**
//	 * Function for setting VolumetricData object on which FeatureExtractor will operate.
//	 * @param volData data on which feature extractor will operate.
//	 */
//	public void setVolData(VolumetricData volData) {
//		this.volData = volData;
//	}
}
