package lv.edi.EDI_3DAtA.bloodvesselsegm;

import java.util.ArrayList;

import boofcv.struct.image.ImageFloat32;

/**
 * 
 * @author Code - Ričards Cacurs.
 * 
 *  Class representing Stacked Multiscale Feature extractor from paper:
 *  Adam Coates and Andrew Y. Ng "Learning Feature Representations with K-means", 
 *  Neural Networks: Tricks of the Trade, 2nd edn, Springer, 2012.
 *
 */
public class SMFeatureExtractor {
	private ArrayList<ImageFloat32> filterBank;

	/**
	 * @return the filterBank
	 */
	public ArrayList<ImageFloat32> getFilterBank() {
		return filterBank;
	}

	/**
	 * Function for setting feature extractor filter-bank from .csv file containing
	 * filter kernel values.
	 * @param filterBank the filterBank to set
	 */
	public void setFilterBank(ArrayList<ImageFloat32> filterBank) {
		this.filterBank = filterBank;
	}
}
