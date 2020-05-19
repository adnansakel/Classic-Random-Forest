#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <map>
#include <random>
#include <ctime>

using namespace cv;
using namespace std;


#define MAX_DEPTH  15
#define FILTER_SIZE 3
#define NUM_TRAINING_SAMPLES 99//99
#define NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE 99
#define IMAGE_DATA_DIMENSION 32//32
#define NUM_RAND_LOC_ON_IMAGE 17//17 good
#define NUM_OF_FILTERS 6
#define NUM_OF_CLASSES 3
#define MIN_SAMPLE 5
#define NUM_OF_TREE 50//50
#define NUM_OF_TEST_SAMPLES 51
#define NUM_OF_THRESHOLD 5
#define ROUGH_NORMALIZATION 1
#define NUM_FEATURE_TYPE 3
#define INTENSITY 0
#define PROBABILITY_ESTIMATE 1
#define PATH 2
#define PATH_LENGTH 2
#define SHIFT_DIFF 3
#define NUM_OF_TREE_IN_PREV_FOREST 10

struct features {
	int class_name = 0;
	int intensity = 0;
	double probability_estimate[NUM_OF_CLASSES];// = 0;
	int path_info_with_leaf_id[NUM_OF_TREE_IN_PREV_FOREST];
	//int path_length = 0;
	//int shift_diff = 0;
};
vector<vector<features>>training_data;
vector<vector<features>>test_data;
//vector<vector<double>>training_data_probability;
//vector<vector<double>>test_data_probability;
//vector<vector<int>>training_data_path_length;
//vector<vector<int>>test_data_path_length;
//vector<vector<int>>training_data_shift_diff;
//vector<vector<int>>test_data_shift_diff;
//vector<int>training_data_labels;
int arr_corr[NUM_OF_CLASSES];
int arr_occur[NUM_OF_CLASSES];
int training_data_dist[NUM_OF_CLASSES];
int feature_type_selection_counter[NUM_FEATURE_TYPE];

int **filter_one; 
int **filter_two; 
int **filter_three; 
int **filter_four; 
int **filter_five;
int **filter_six;


int max_response = -9999999;//for debugging/testing
int min_response = 9999999;//for debugging/testing
long long rand_seed;
map<int, int>global_response;//for debugging/testing

default_random_engine generator(getTickCount());
normal_distribution<double> row_distribution(16, 12);//maen = 256/2; std = 32.0
int normal[29][29];
int tree_node_count = 0;
int tree_height = -1;
int largest_leaf_with_data_samples = 0;
double forest_balance = 0;
double forest_balance_ = 0;
double forest_saturation = 0;

class Node {
	//int histogram[10];//holds the number of samples in each class.
public:
	Node(){}
	Node *Left, *Right;
	int index;// will be removed later
	int imleaf = 0;// indicates a leaf node
	vector<int> samples;//stores the index of splitted data samples
	int feature_index;//image index that gave maximum information gain or minimum gini index
	int **feature_filter;
	vector<int> histogram;
	int threshold;
	int impurity = 1;
	//map<int, int> histogram_filter_response;
	double classProbability[NUM_OF_CLASSES];
	int feature_type = 0;
	int class_name_for_probability = 0;
	int tcr_threshold_sample_index;
};
struct SplittedSamples {
	vector<int>left_samples;//holds indexes of the data
	vector<int>right_samples;
	vector<int> left_histogram;
	vector<int> right_histogram;
	int threshold;
	int class_name_for_probability = 0;
	//map<int,int> histogram_filter_response;
	double gini;
	int **filter;
	int feature_type;
	int tcr_threshold_sample_index;
};

struct Samples {
	int sample_index;
	int classname;
}sortedSamples[NUM_TRAINING_SAMPLES];

bool comp(struct Samples a, struct Samples b) {
	return a.classname < b.classname;
}
/*void split(Node *node, int depth, int index) {
	if (depth == 4) return;
	node->index = index;
	node->Left = split()
}*/
double getGini(vector<int> left_histogram, vector<int> right_histogram, int left_sample_size, int right_sample_size) {
	double gini_l = 0;
	double gini_r = 0;
	for (int i = 0; i < left_histogram.size(); i++) {
		if (left_sample_size != 0) { gini_l += ((double)left_histogram[i] / left_sample_size)*((double)1 - (double)left_histogram[i] / left_sample_size); }
		if (right_sample_size != 0) { gini_r += ((double)right_histogram[i] / right_sample_size)*((double)1 - (double)right_histogram[i] / right_sample_size); }
	}
	if (left_sample_size + right_sample_size != 0) {
		gini_l = gini_l*((double)left_sample_size / (left_sample_size + right_sample_size));
		gini_r = gini_r*((double)right_sample_size / (left_sample_size + right_sample_size));
	}
	return gini_l+gini_r;
}

/*
Following method considers histogram got from the filter response on each image for different points on image
*/
double getGini(map<int,int> &histogram_left, map<int, int> &histogram_right, int size_left, int size_right) {
	double gini_l = 0;
	double gini_r = 0;
	double p_l, p_r;
	//cout << "Histogram Left:" << endl;
	/*
	for (int i = 0; i < 36000 / ROUGH_NORMALIZATION; i++) {
		//cout << hist->first << ": " << hist->second << endl;
		//if (hist->first == 0)continue;
		if (size_left != 0) {
			p_l = (double)histogram_left[i] / (size_left);
			gini_l += p_l*(1 - p_l);
		}

		if (size_right != 0) {
			p_r = (double)histogram_right[i] / (size_right);

			gini_r += p_r*(1 - p_r);
		}
		
	}
	*/
	
	for (map<int, int>::iterator hist = histogram_left.begin(); hist != histogram_left.end(); ++hist) {
		if (size_left != 0) {
			p_l = (double)hist->second / (size_left);
			gini_l += p_l*(1 - p_l);
		}
	}
	
	for (map<int, int>::iterator hist = histogram_right.begin(); hist != histogram_right.end(); ++hist) {
		if (size_right != 0) {
			p_r = (double)hist->second / (size_right);
			gini_r += p_r*(1 - p_r);
		}
	}
	
	if (size_left + size_right != 0) {
		gini_l = gini_l*((double)size_left / (size_left + size_right));
		gini_r = gini_r*((double)size_right / (size_left + size_right));
	}

	return gini_l+gini_r;
}
void populateTextonHistForestTest(int sample_index, int filter_position, int feature_type_path, vector<map<int, int>>&texton_hist_forest) {
	map<int, int> P_texton_hist;
	vector<int> unique_nodes;
	double response = 0;
	int row = filter_position / IMAGE_DATA_DIMENSION;
	int col = filter_position % IMAGE_DATA_DIMENSION;
	//int t = 0;
	int len_p = 0;
	int node;
	//int node_number;
	map<int, int>texton_hist_tree;

	for (int t = 0; t < NUM_OF_TREE; t++) {
		P_texton_hist.clear();
		texton_hist_tree.clear();

		unique_nodes.clear();
		//row_p = P.pixel_position / image_size;
		//col_p = P.pixel_position % image_size;

		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= test_data[sample_index].size()) { continue; }
				node = test_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].path_info_with_leaf_id[t];
				if (P_texton_hist[node] == 0) {
					unique_nodes.push_back(node);
				}
				P_texton_hist[node]++;
			}
			//cout << endl;
		}

		for (int n = 0; n < unique_nodes.size(); n++) {
			texton_hist_tree[unique_nodes[n]] = P_texton_hist[unique_nodes[n]];

			//texton_hist_tree.push_back(mp);
		}

		texton_hist_forest.push_back(texton_hist_tree);

	}

}

void populateTextonHistForest(int sample_index, int filter_position, int feature_type_path, vector<map<int, int>>&texton_hist_forest) {
	map<int, int> P_texton_hist;
	vector<int> unique_nodes;
	double response = 0;
	int row = filter_position / IMAGE_DATA_DIMENSION;
	int col = filter_position % IMAGE_DATA_DIMENSION;
	//int t = 0;
	int len_p = 0;
	int node;
	//int node_number;
	map<int, int>texton_hist_tree;

	for (int t = 0; t < NUM_OF_TREE; t++) {
		P_texton_hist.clear();
		texton_hist_tree.clear();

		unique_nodes.clear();
		//row_p = P.pixel_position / image_size;
		//col_p = P.pixel_position % image_size;

		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= training_data[sample_index].size()) { continue; }
				node = training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].path_info_with_leaf_id[t];
				if (P_texton_hist[node] == 0) {
					unique_nodes.push_back(node);
				}
				P_texton_hist[node]++;
			}
			//cout << endl;
		}

		for (int n = 0; n < unique_nodes.size(); n++) {
			texton_hist_tree[unique_nodes[n]] = P_texton_hist[unique_nodes[n]];

			//texton_hist_tree.push_back(mp);
		}

		texton_hist_forest.push_back(texton_hist_tree);

	}

}
double getTextonHistogramComparisonWithLeafID(vector<map<int, int>>&P_texton_hist_f, vector<map<int, int>>&Q_texton_hist_f) {
	if (P_texton_hist_f.size() == 0 || Q_texton_hist_f.size() == 0) { return 0; }
	double response = 0;
	double h_insertion = 0;
	int hist_size;
	double sum = 0;
	for (int t = 0; t < NUM_OF_TREE; t++) {
		h_insertion = 0;
		hist_size = 0;

		for (map<int, int>::iterator hist = P_texton_hist_f[t].begin(); hist != P_texton_hist_f[t].end(); ++hist) {
			hist_size++;// = hist->second;
						//cout << hist->first << " " << hist->second << " " << Q_texton_hist_f[t][hist->first] << endl;
			if (hist->second < Q_texton_hist_f[t][hist->first]) {
				//cout << "HHHH!!" << endl;
				h_insertion += hist->second;
			}
			else {
				//cout << "HHHH!!!" << endl;
				h_insertion += Q_texton_hist_f[t][hist->first];
			}
		}

		if (hist_size > 0) { sum += h_insertion / (hist_size*FILTER_SIZE*FILTER_SIZE); }

	}

	if (sum > 0) { response = sum / NUM_OF_TREE * 100; }
	//cout << response << " "<<endl;
	return response;
}

int getFilteredResponse(int sample_index, int filter_position, int **filter, int feature_type_prob, int class_name) {
	//sample_index tells which image training data to load
	//filter_position tells where to put the top left corner of the filter on image data which is a 1d vector
	//**filter takes the desired filter be used
	int res = 0;
	int row = filter_position / IMAGE_DATA_DIMENSION;
	int col = filter_position % IMAGE_DATA_DIMENSION;

	if (feature_type_prob == PROBABILITY_ESTIMATE) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= training_data[sample_index].size()) { continue; }
				res += ((int)((training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].probability_estimate[class_name]) * 10000)) * filter[i][j];
				//cout << training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)] * filter[i][j] << " ";
			}
			//cout << endl;
		}
	}

	return (int)(round((double)res / ROUGH_NORMALIZATION));
}
int getFilteredResponse(int sample_index, int filter_position, int **filter, int feature_type) {
	//sample_index tells which image training data to load
	//filter_position tells where to put the top left corner of the filter on image data which is a 1d vector
	//**filter takes the desired filter be used
	int res = 0;
	int row = filter_position / IMAGE_DATA_DIMENSION;
	int col = filter_position % IMAGE_DATA_DIMENSION;
	
	
	if(feature_type == INTENSITY){
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= training_data[sample_index].size()) { continue; }
				res += training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].intensity * filter[i][j];
				//cout << training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)] * filter[i][j] << " ";
			}
			//cout << endl;
		}
	}

	/*
	else if (feature_type == PROBABILITY_ESTIMATE) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= training_data[sample_index].size()) { continue; }
				res += ((int)((training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].probability_estimate) * 10000)) * filter[i][j];
				//cout << training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)] * filter[i][j] << " ";
			}
			//cout << endl;
		}
	}
	
	else if (feature_type == PATH_LENGTH) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= training_data[sample_index].size()) { continue; }
				res += training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].path_length * filter[i][j];
				//cout << training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)] * filter[i][j] << " ";
			}
			//cout << endl;
		}
	}
	else {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= training_data[sample_index].size()) { continue; }
				res += training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].shift_diff * filter[i][j];
				//cout << training_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)] * filter[i][j] << " ";
			}
			//cout << endl;
		}
	}

	*/
	

	//if (res == 0)return 0;
	return (int)(round((double)res/ ROUGH_NORMALIZATION));
}


int getFilteredResponseTesting(int sample_index, int filter_position, int **filter, int featur_type_prob, int class_name) {
	//sample_index tells which image training data to load
	//filter_position tells where to put the top left corner of the filter on image data which is a 1d vector
	//**filter takes the desired filter be used
	int res = 0;
	int row = filter_position / IMAGE_DATA_DIMENSION;
	int col = filter_position % IMAGE_DATA_DIMENSION;

	if (featur_type_prob == PROBABILITY_ESTIMATE) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= test_data[sample_index].size()) { continue; }
				res += ((int)((test_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].probability_estimate[class_name]) * 10000)) * filter[i][j];
			}
		}
	}

	return (int)(round((double)res / ROUGH_NORMALIZATION));
}

int getFilteredResponseTesting(int sample_index, int filter_position, int **filter, int featur_type) {
	//sample_index tells which image training data to load
	//filter_position tells where to put the top left corner of the filter on image data which is a 1d vector
	//**filter takes the desired filter be used
	int res = 0;
	int row = filter_position / IMAGE_DATA_DIMENSION;
	int col = filter_position % IMAGE_DATA_DIMENSION;

	if (featur_type == INTENSITY) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= test_data[sample_index].size()) { continue; }
				res += test_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].intensity * filter[i][j];
			}
		}
	}
	/*
	else if (featur_type == PROBABILITY_ESTIMATE) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= test_data[sample_index].size()) { continue; }
				res += ((int)((test_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].probability_estimate)*10000)) * filter[i][j];
			}
		}
	}
	else if (featur_type == PATH_LENGTH) {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= test_data[sample_index].size()) { continue; }
				res += test_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].path_length * filter[i][j];
			}
		}
	}
	else {
		for (int i = 0; i < FILTER_SIZE; i++) {
			for (int j = 0; j < FILTER_SIZE; j++) {
				if ((row + i)*IMAGE_DATA_DIMENSION + (col + j) >= test_data[sample_index].size()) { continue; }
				res += test_data[sample_index][(row + i)*IMAGE_DATA_DIMENSION + (col + j)].shift_diff * filter[i][j];
			}
		}
	}
	*/
	//if (res == 0)return 0;
	return (int)(round((double)res / ROUGH_NORMALIZATION));
}


SplittedSamples getSpittedSampleswithGini(int** filter, int filter_loc, vector<int>samples_index) {
	
	struct SplittedSamples chosen_samples;
	map<int, int>histogram_filter_response_left;
	map<int, int>histogram_filter_response_right;
	histogram_filter_response_left.clear();
	histogram_filter_response_right.clear();
	vector<int> filter_response;
	vector<int> unique_filter_response;
	map<int,int>is_it_unique;
	vector<map<int, int>> P_hist;
	vector<map<int, int>> Q_hist;
	int count_left = 0;
	int count_right = 0;
	double min_gini = 99999;
	//cout << "filter position: " << filter_loc<<" "<<filter_loc/IMAGE_DATA_DIMENSION <<","<<filter_loc%IMAGE_DATA_DIMENSION<<endl;
	
	//int freq_resp_max = -1;
	//int freq_resp_max_val = -5;
	for (int f = 0; f < NUM_FEATURE_TYPE; f++) {
		//if (f == 2 || f == 3) { continue; }//Only wan to test for intensity and path length
		int bad_loc_count = 0;
		if (f == 1) {
			for (int cl = 0; cl < NUM_OF_CLASSES; cl++) {
				is_it_unique.clear();
				filter_response.clear();
				unique_filter_response.clear();
				//global_response.clear();
				for (int i = 0; i < samples_index.size(); i++) {
					filter_response.push_back(getFilteredResponse(samples_index.at(i), filter_loc, filter, f, cl));
					if (is_it_unique[filter_response[i]] == 0) { unique_filter_response.push_back(filter_response[i]); }
					is_it_unique[filter_response[i]] = 1;
					//global_response[filter_response[i]]++;

				}

				//cout << "resp: " << freq_resp_max_val << " freq: " << freq_resp_max << endl;
				rand_seed = getTickCount();
				//rand_seed += 123456789;
				//cout << "Choosing threshold " << rand_seed << endl;
				//srand(rand_seed);
				int num_of_threshold = (int)sqrt((double)unique_filter_response.size());
				//num_of_threshold = unique_filter_response.size();
				for (int k = 0; k < num_of_threshold; k++) {
					int threshold_index = rand() % unique_filter_response.size();
					int threshold = unique_filter_response[threshold_index];

					//cout << "Num of unique responses: "<< unique_filter_response.size()<<" Threshold index:"<<threshold_index<<" Threshold: " << threshold << endl;
					struct SplittedSamples samples;
					for (int i = 0; i < NUM_OF_CLASSES; i++) {
						samples.left_histogram.push_back(0);
						samples.right_histogram.push_back(0);
					}
					count_left = 0;
					count_right = 0;
					histogram_filter_response_left.clear();
					histogram_filter_response_right.clear();
					for (int i = 0; i < samples_index.size(); i++) {

						//cout << "response:" << filter_response[i] << endl;
						samples.threshold = threshold;
						if (filter_response[i] < threshold) {
							//if (filter_response != 0) {
							histogram_filter_response_left[filter_response[i]]++;
							count_left++;
							//}

							samples.left_samples.push_back(samples_index.at(i));
							samples.left_histogram.at(training_data[samples_index.at(i)][0].class_name)++;
							samples.filter = filter;
						}
						else {
							//if (filter_response != 0) {
							histogram_filter_response_right[filter_response[i]]++;
							count_right++;
							//}
							samples.right_samples.push_back(samples_index.at(i));
							samples.right_histogram.at(training_data[samples_index.at(i)][0].class_name)++;
							samples.filter = filter;

						}
					}

					//samples.gini = getGini(histogram_filter_response_left, histogram_filter_response_right, count_left, count_right);//passing feature label histogram
					samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());//passing class label histogram
																																					   //cout << "left: " << count_left << " Right: " << count_right << " gini: "<<samples.gini<<endl;
					if (samples.gini < min_gini) {
						min_gini = samples.gini;
						chosen_samples = samples;
						chosen_samples.gini = min_gini;
						chosen_samples.feature_type = f;
						chosen_samples.class_name_for_probability = cl;
					}
				}
			}
			
		}
		else if(f==0){
			is_it_unique.clear();
			filter_response.clear();
			unique_filter_response.clear();
			//global_response.clear();
			for (int i = 0; i < samples_index.size(); i++) {
				filter_response.push_back(getFilteredResponse(samples_index.at(i), filter_loc, filter, f));
				if (is_it_unique[filter_response[i]] == 0) { unique_filter_response.push_back(filter_response[i]); }
				is_it_unique[filter_response[i]] = 1;
				//global_response[filter_response[i]]++;

			}

			//cout << "resp: " << freq_resp_max_val << " freq: " << freq_resp_max << endl;
			rand_seed = getTickCount();
			//rand_seed += 123456789;
			//cout << "Choosing threshold " << rand_seed << endl;
			//srand(rand_seed);
			int num_of_threshold = (int)sqrt((double)unique_filter_response.size());
			//num_of_threshold = unique_filter_response.size();
			for (int k = 0; k < num_of_threshold; k++) {
				int threshold_index = rand() % unique_filter_response.size();
				int threshold = unique_filter_response[threshold_index];

				//cout << "Num of unique responses: "<< unique_filter_response.size()<<" Threshold index:"<<threshold_index<<" Threshold: " << threshold << endl;
				struct SplittedSamples samples;
				for (int i = 0; i < NUM_OF_CLASSES; i++) {
					samples.left_histogram.push_back(0);
					samples.right_histogram.push_back(0);
				}
				count_left = 0;
				count_right = 0;
				histogram_filter_response_left.clear();
				histogram_filter_response_right.clear();
				for (int i = 0; i < samples_index.size(); i++) {

					//cout << "response:" << filter_response[i] << endl;
					samples.threshold = threshold;
					if (filter_response[i] < threshold) {
						//if (filter_response != 0) {
						histogram_filter_response_left[filter_response[i]]++;
						count_left++;
						//}

						samples.left_samples.push_back(samples_index.at(i));
						samples.left_histogram.at(training_data[samples_index.at(i)][0].class_name)++;
						samples.filter = filter;
					}
					else {
						//if (filter_response != 0) {
						histogram_filter_response_right[filter_response[i]]++;
						count_right++;
						//}
						samples.right_samples.push_back(samples_index.at(i));
						samples.right_histogram.at(training_data[samples_index.at(i)][0].class_name)++;
						samples.filter = filter;

					}
				}

				//samples.gini = getGini(histogram_filter_response_left, histogram_filter_response_right, count_left, count_right);//passing feature label histogram
				samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());//passing class label histogram
																																				   //cout << "left: " << count_left << " Right: " << count_right << " gini: "<<samples.gini<<endl;
				if (samples.gini < min_gini) {
					min_gini = samples.gini;
					chosen_samples = samples;
					chosen_samples.gini = min_gini;
					chosen_samples.feature_type = f;
				}
			}
		}
		else {
			if(!(filter[0][0]==2 && filter[0][1] == 2 && filter[0][2] == 2 && 
				filter[1][0] == 2 && filter[1][1] == 2 && filter[1][2] == 2 &&
				filter[2][0] == 2 && filter[2][1] == 2 && filter[2][2] == 2)) {
				continue;//making sure no extra calculation is done for multiple filters
			}
			is_it_unique.clear();
			filter_response.clear();
			unique_filter_response.clear();
			double tcr = 0;
			//global_response.clear();
			int rand_samples = (int)sqrt((double)samples_index.size());
			for (int rs = 0; rs < rand_samples; rs++) {
				is_it_unique.clear();
				filter_response.clear();
				unique_filter_response.clear();
				for (int i = 0; i < samples_index.size(); i++) {
					P_hist.clear(); P_hist.shrink_to_fit(); Q_hist.clear(); Q_hist.shrink_to_fit();
					populateTextonHistForest(samples_index[rs], filter_loc, PATH, P_hist);
					populateTextonHistForest(samples_index[i], filter_loc, PATH, Q_hist);
					tcr = getTextonHistogramComparisonWithLeafID(P_hist, Q_hist);
					//unique_filter_response.push_back();

					filter_response.push_back(tcr);
					if (is_it_unique[filter_response[i]] == 0) { unique_filter_response.push_back(filter_response[i]); }
					is_it_unique[filter_response[i]] = 1;
					//global_response[filter_response[i]]++;

				}

				//cout << "resp: " << freq_resp_max_val << " freq: " << freq_resp_max << endl;
				rand_seed = getTickCount();
				//rand_seed += 123456789;
				//cout << "Choosing threshold " << rand_seed << endl;
				//srand(rand_seed);
				int num_of_threshold = (int)sqrt((double)unique_filter_response.size());
				//num_of_threshold = unique_filter_response.size();
				for (int k = 0; k < num_of_threshold; k++) {
					int threshold_index = rand() % unique_filter_response.size();
					int threshold = unique_filter_response[threshold_index];

					//cout << "Num of unique responses: "<< unique_filter_response.size()<<" Threshold index:"<<threshold_index<<" Threshold: " << threshold << endl;
					struct SplittedSamples samples;
					for (int i = 0; i < NUM_OF_CLASSES; i++) {
						samples.left_histogram.push_back(0);
						samples.right_histogram.push_back(0);
					}
					count_left = 0;
					count_right = 0;
					histogram_filter_response_left.clear();
					histogram_filter_response_right.clear();
					for (int i = 0; i < samples_index.size(); i++) {

						//cout << "response:" << filter_response[i] << endl;
						samples.threshold = threshold;
						samples.tcr_threshold_sample_index = samples_index[rs];
						if (filter_response[i] < threshold) {
							//if (filter_response != 0) {
							histogram_filter_response_left[filter_response[i]]++;
							count_left++;
							//}

							samples.left_samples.push_back(samples_index.at(i));
							samples.left_histogram.at(training_data[samples_index.at(i)][0].class_name)++;
							samples.filter = filter;
						}
						else {
							//if (filter_response != 0) {
							histogram_filter_response_right[filter_response[i]]++;
							count_right++;
							//}
							samples.right_samples.push_back(samples_index.at(i));
							samples.right_histogram.at(training_data[samples_index.at(i)][0].class_name)++;
							samples.filter = filter;

						}
					}

					//samples.gini = getGini(histogram_filter_response_left, histogram_filter_response_right, count_left, count_right);//passing feature label histogram
					samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());//passing class label histogram
																																					   //cout << "left: " << count_left << " Right: " << count_right << " gini: "<<samples.gini<<endl;
					if (samples.gini < min_gini) {
						min_gini = samples.gini;
						chosen_samples = samples;
						chosen_samples.gini = min_gini;
						chosen_samples.feature_type = f;
					}
				}
			}
			
		}
	}

	
	
	//samples.gini = getGini(samples.left_histogram, samples.right_histogram, samples.left_samples.size(), samples.right_samples.size());
	
	//cout << "Left Sample size: " << samples.left_samples.size() << " right sample size: " << samples.right_samples.size() <<" Gini:"<<samples.gini<< endl;
	return chosen_samples;
}

Node* BuildTree( Node* node, int depth, vector<int>samples_index, vector<int>histogram, int impurity) {
	/*
	This function builds a tree with training samples
	*/
	vector<int>left_child_saple_index;
	vector<int>right_child_saple_index;
	tree_node_count++;
	if (depth > tree_height) { tree_height = depth; }
	if (node == NULL) {
		node = new Node();
	}

	if (depth == MAX_DEPTH) {
		node->samples = samples_index;
		node->histogram = histogram;
		for (int k = 0; k < NUM_OF_CLASSES; k++) {
			node->classProbability[k] = (double)node->histogram[k] / node->samples.size();
		}
		node->imleaf = 1;
		if (node->samples.size() > largest_leaf_with_data_samples) { largest_leaf_with_data_samples = node->samples.size(); }
		return node; 
	}
	if (samples_index.size() < MIN_SAMPLE) { 
		node->samples = samples_index;
		node->histogram = histogram;
		for (int k = 0; k < NUM_OF_CLASSES; k++) {
			node->classProbability[k] = (double)node->histogram[k] / node->samples.size();
		}
		node->imleaf = 1;
		if (node->samples.size() > largest_leaf_with_data_samples) { largest_leaf_with_data_samples = node->samples.size(); }
		return node;
	}
	node->impurity = impurity;
	//node->index = index;
	//cout << node->index << endl;
	/*
		Check for spittiing feature interms of filter and image_index based on filter response
		Randomly choose five location on image, place each filter on each of the location, split the dataset
		based on the filter response. Calculate the gini index for each of the filter and index pairs for the entire training dataset.
		Choose the one with minimum gini index.
	*/
	double gini = 99999;
	int chosen_threshold = 0;
	struct SplittedSamples splitted_samples;
	struct SplittedSamples testsamples;
	//rand_seed += 123456789;
	//cout << "Building tree " << rand_seed << endl;
	//srand(rand_seed);
	int row, col;
		//srand(getTickCount());
		for (int j = 0; j < NUM_RAND_LOC_ON_IMAGE; j++) {
			int filter_position_on_image;// = 1 + rand() % 784;
			//filter_position_on_image = 1 + rand() % 784;
			row = (int)row_distribution(generator);
			col = (int)row_distribution(generator);
			if (row < 0) { row = 0; }
			else if (row > IMAGE_DATA_DIMENSION-1) { row = IMAGE_DATA_DIMENSION-1; }
			if (col < 0) { col = 0; }
			else if (col > IMAGE_DATA_DIMENSION - 1) { col = IMAGE_DATA_DIMENSION-1; }
			filter_position_on_image = row*IMAGE_DATA_DIMENSION + col;
			//int i = rand() % NUM_OF_FILTERS;
			//double row = (int)row_distribution(generator);
			for (int i = 0; i < NUM_OF_FILTERS; i++) {
				if (i == 0) {
					testsamples = getSpittedSampleswithGini(filter_one, filter_position_on_image, samples_index);

				}
				else if (i == 1) {
					testsamples = getSpittedSampleswithGini(filter_two, filter_position_on_image, samples_index);
				}
				else if (i == 2) {
					testsamples = getSpittedSampleswithGini(filter_three, filter_position_on_image, samples_index);
				}
				else if(i == 3){
					testsamples = getSpittedSampleswithGini(filter_four, filter_position_on_image, samples_index);
				}
				else if(i == 4){
					testsamples = getSpittedSampleswithGini(filter_five, filter_position_on_image, samples_index);
				}
				else {
					testsamples = getSpittedSampleswithGini(filter_six, filter_position_on_image, samples_index);
				}
				
				//cout << "Left samples size: " << testsamples.left_samples.size() << "Right samples size: " << testsamples.right_samples.size() << " "<<"gini: "<<testsamples.gini<<endl;
				if (testsamples.gini < gini) {
					gini = testsamples.gini;
					splitted_samples = testsamples;
					chosen_threshold = testsamples.threshold;
					node->feature_index = filter_position_on_image;
					node->feature_type = testsamples.feature_type;
					node->class_name_for_probability = testsamples.class_name_for_probability;
					node->tcr_threshold_sample_index = splitted_samples.tcr_threshold_sample_index;
				}
			
		}
	}

	//	if (node->impurity < gini) { node->imleaf = 1; return node; }
		feature_type_selection_counter[node->feature_type]++;
	node->feature_filter = splitted_samples.filter;
	node->threshold = chosen_threshold;
	node->samples = samples_index;
	node->histogram = histogram;
	for (int k = 0; k < NUM_OF_CLASSES; k++) {
		node->classProbability[k] = (double)node->histogram[k] / node->samples.size();
	}
	cout << "chosen leftsamplesize: " << splitted_samples.left_samples.size() << " chosen rightsamplesize: " << splitted_samples.right_samples.size() << " Gini: "<<gini<<endl;
	
	if (splitted_samples.left_samples.size() == 0 || splitted_samples.right_samples.size() == 0) {
		node->imleaf = 1;
		if (node->samples.size() > largest_leaf_with_data_samples) { largest_leaf_with_data_samples = node->samples.size(); }
		return node;
	}
	if (splitted_samples.left_samples.size() > 0) {
		node->Left = BuildTree(node->Left, depth + 1, splitted_samples.left_samples, splitted_samples.left_histogram, gini);
	}
	if (splitted_samples.right_samples.size() > 0) {
		node->Right = BuildTree(node->Right, depth + 1, splitted_samples.right_samples, splitted_samples.right_histogram,gini);
	}
	
	/*
	if (node->Left == NULL && node->Right == NULL) {
		node->imleaf = 1;
	}*/
	return node;
}

double* getPredictionHistogram(Node* node, int sample_index) {
	Node* nextRightNode = node->Right;
	Node* nextLeftNode = node->Left;
	Node* currNode = node;
	vector<map<int, int>> P_hist;
	vector<map<int, int>> Q_hist;
	double tcr = 0;
	/*
	cout << nextNode->index <<" "<<node->index<< endl;
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	nextNode = nextNode->Right;
	cout << nextNode->index << " " << node->index << endl;
	if (nextNode->Right != NULL) {
		nextNode = nextNode->Right;
		cout << nextNode->index << " " << node->index << endl;
	}*/
	
		
	while (1) {
		//cout << currNode->index << endl;
		//if (currNode == NULL) { break; }
		if (currNode->imleaf) { 
			break; 
		}

		Node* nextRightNode = currNode->Right;
		Node* nextLeftNode = currNode->Left;
		if (currNode->feature_type == PROBABILITY_ESTIMATE) {
			if (getFilteredResponseTesting(sample_index, currNode->feature_index, currNode->feature_filter, currNode->feature_type, currNode->class_name_for_probability) < currNode->threshold) {//<threshold
				currNode = nextLeftNode;
			}
			else {

				currNode = nextRightNode;
			}
		}
		else if (currNode->feature_type == PATH) {
			//cout << "Here!" << endl;
			P_hist.clear(); P_hist.shrink_to_fit(); Q_hist.clear(); Q_hist.shrink_to_fit();
			populateTextonHistForest(currNode->tcr_threshold_sample_index, currNode->feature_index, PATH, P_hist);
			populateTextonHistForestTest(sample_index, currNode->feature_index, PATH, Q_hist);
			tcr = getTextonHistogramComparisonWithLeafID(P_hist, Q_hist);
			//cout << "Here!!" << endl;
			if (tcr < currNode->threshold) {//<threshold
				currNode = nextLeftNode;
			}
			else {

				currNode = nextRightNode;
			}
		}
		else {
			if (getFilteredResponseTesting(sample_index, currNode->feature_index, currNode->feature_filter, currNode->feature_type) < currNode->threshold) {//<threshold
				currNode = nextLeftNode;
			}
			else {

				currNode = nextRightNode;
			}
		}
		
		
		
	}
	//for (int i = 0; i < NUM_OF_CLASSES; i++) { cout << currNode->classProbability[i] << " "; }cout << endl;
	return currNode->classProbability;
}

void populateFilters() {

	filter_one = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_two = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_three = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_four = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_five = (int**)malloc(FILTER_SIZE * sizeof(int*));
	
	
	for (int i = 0; i < FILTER_SIZE; i++) {
		filter_one[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_two[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_three[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_four[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_five[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		

	}

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (i < FILTER_SIZE/2) {
				filter_one[i][j] = 1;
			}
			else {
				filter_one[i][j] = 2;
			}
			cout << filter_one[i][j];
		}
		cout << endl;
	}

	cout << endl << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (j < FILTER_SIZE / 3 || j > (FILTER_SIZE*2 / 3)-1) {
				filter_two[i][j] = 1;
			}
			else {
				filter_two[i][j] = 2;
			}
			cout << filter_two[i][j];
		}
		cout << endl;
	}

	cout << endl << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (j > FILTER_SIZE/2 -1) {
				filter_three[i][j] = 1;
			}
			else {
				filter_three[i][j] = 2;
			}
			cout << filter_three[i][j];
		}
		cout << endl;
	}

	cout << endl << endl;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			if (i < FILTER_SIZE / 2 && j < FILTER_SIZE / 2) {
				filter_four[i][j] = 1;
			}
			else if(i > (FILTER_SIZE / 2 - 1) && j > (FILTER_SIZE / 2 - 1 )){
				filter_four[i][j] = 1;
			}
			else {
				filter_four[i][j] = 2;
			}
			cout << filter_four[i][j];
		}
		cout << endl;
	}

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			filter_five[i][j] = 1;
			cout << filter_five[i][j];
		}
		cout << endl;
	}
	cout << endl;
}
void populate_3_by_3_filters() {
	filter_one = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_two = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_three = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_four = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_five = (int**)malloc(FILTER_SIZE * sizeof(int*));
	filter_six = (int**)malloc(FILTER_SIZE * sizeof(int*));


	for (int i = 0; i < FILTER_SIZE; i++) {
		filter_one[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_two[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_three[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_four[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_five[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));
		filter_six[i] = (int*)malloc(FILTER_SIZE * sizeof(int*));


	}

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			
				filter_one[i][j] = 2;
				cout << filter_one[i][j];
		}
		cout << endl;
		
	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if ((i*FILTER_SIZE + j) % 2 == 0) {
				filter_two[i][j] = 2;

			}
			else {
				filter_two[i][j] = 1;
			}
			cout << filter_two[i][j];
		}
		cout << endl;
	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (i%2 == 1 || j%2 == 1) {
				filter_three[i][j] = 2;

			}
			else {
				filter_three[i][j] = 1;
			}
			cout << filter_three[i][j];
		}
		cout << endl;

	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (i == 1) {
				filter_four[i][j] = 2;

			}
			else {
				filter_four[i][j] = 1;
			}
			cout << filter_four[i][j];
		}
		cout << endl;

	}
	cout << endl;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {

			if (j == 1) {
				filter_five[i][j] = 2;

			}
			else {
				filter_five[i][j] = 1;
			}
			cout << filter_five[i][j];
		}
		cout << endl;
	}

	cout << endl;
	filter_six[0][0] = 1; //filter_six[0][1] = 1; filter_six[0][2] = 2;
	//filter_six[1][0] = 1; filter_six[1][1] = 2; filter_six[1][2] = 1;
	//filter_six[2][0] = 2; filter_six[2][1] = 1; filter_six[2][2] = 1;

}
/*
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr)
{
	arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
	ifstream file("D:\\data0", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i<number_of_images; ++i)
		{
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					arr[i][(n_rows*r) + c] = (double)temp;
				}
			}
		}
	}
}
*/

void readMNSITfromCSV(string filepath, vector<vector<int>> &data_samples, int num_of_samples) {
	
	/*
	int random_sample_index[NUM_TRAINING_SAMPLES];
	for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
		random_sample_index[i] = rand() % 600000;//60000 is the total data sample in origina training data.
	}*/
	ifstream inputfle(filepath);
	string current_line; 
	int num_line_read = 1;
	while (getline(inputfle, current_line)) {
		// Now inside each line we need to seperate the cols
		//cout << "test";
		if (num_line_read > num_of_samples) break;
		vector<int> values;
		stringstream temp(current_line);
		string single_value;
		while (getline(temp, single_value, ',')) {
			// convert the string element to a integer value
			values.push_back(atoi(single_value.c_str()));
			cout << atoi(single_value.c_str()) << " ";
		}
		cout << endl;
		// add the row to the complete data vector
		data_samples.push_back(values);
		num_line_read++;
	}
	//cout << num_line_read << endl;
}

/*void maptest(map<int, int> &mymap) {
	for (map<int, int>::iterator it = mymap.begin(); it != mymap.end(); ++it)
		std::cout << it->first << " => " << it->second << '\n';
}*/

void printParameters() {
		cout << "MAX_DEPTH " << MAX_DEPTH << endl;
		cout << "FILTER_SIZE " << FILTER_SIZE << endl;
		cout << "NUM_TRAINING_SAMPLES "<< NUM_TRAINING_SAMPLES << endl;
		cout << "NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE "<< NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE << endl;
		cout << "IMAGE_DATA_DIMENSION "<< IMAGE_DATA_DIMENSION << endl;
		cout << "NUM_RAND_LOC_ON_IMAGE "<< NUM_RAND_LOC_ON_IMAGE << endl;
		cout << "NUM_OF_FILTERS "<< NUM_OF_FILTERS << endl;
		cout << "NUM_OF_CLASSES "<< NUM_OF_CLASSES << endl;
		cout << "MIN_SAMPLE "<< MIN_SAMPLE << endl;
		cout << "NUM_OF_TREE "<< NUM_OF_TREE << endl;
		cout << "NUM_OF_TEST_SAMPLES "<< NUM_OF_TEST_SAMPLES << endl;
		cout << "NUM_OF_THRESHOLD " << NUM_OF_THRESHOLD << endl;
		cout << "ROUGH_NORMALIZATION " << ROUGH_NORMALIZATION << endl;

}

int two_pow[MAX_DEPTH + 1] = { 1,2,4,8,16,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
int binary_string_to_int(string str) {
	//cout << str << endl;
	int num = 0;
	int len = str.length();
	for (int i = 0; i < len; i++) {
		num += (str[len - 1 - i] - 48)*two_pow[i];
		//cout << (str[len - 1 - i] - 48)*two_pow[i] << endl;
	}
	return num;
}
void readTrainingDatafromText(vector<vector<features>> &data_samples) {
	freopen("training_data_scene_scaled.txt", "r", stdin); 
	//freopen("input_for_pooling_prob.txt", "r", stdin); 
	int classname;
	vector<features> values;
	struct features feature;
	int intensity;
	double probability;
	int index = 0;
	int count = 0;
	//cout << "gggg" << endl;

	

	for (int s = 0; s < NUM_TRAINING_SAMPLES; s++) {
		scanf("%d", &classname);
		//while (scanf("%d", &classname)==1) {
		//count++;
		//cout << "dddd" << endl;

		//cout << classname<<" ";
		feature.class_name = classname;
		values.clear();
		values.push_back(feature);
		training_data_dist[classname]++;
		sortedSamples[index].classname = classname;
		sortedSamples[index].sample_index = index;
		index++;
		//feature.class_name = classname;
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &feature.intensity);
			//feature.intensity = intensity;

			values.push_back(feature);
			//cout << intensity << " ";

		}
		//cout << endl;
		data_samples.push_back(values);
		//if(i==10)break;

		//}
	}
	
	for (int s = 0; s < NUM_TRAINING_SAMPLES; s++) {

		for (int k = 0; k < NUM_OF_CLASSES; k++) {
			scanf("%lf", &probability);
		}
		
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			//cout << "asdsa" <<i<< endl;
			//if (i + 1 == data_samples[s].size()-1)break;
			for (int k = 0; k < NUM_OF_CLASSES; k++) {
				scanf("%lf", &data_samples[s][i + 1].probability_estimate[k]);
			}
			//cout << "asdsa" <<i<< endl;

		}
		//cout << "data loaded" << endl;
	}

	string path;
	for (int s = 0; s < NUM_TRAINING_SAMPLES; s++) {

		/*
		for (int k = 0; k < NUM_OF_TREE_IN_PREV_FOREST; k++) {
			//scanf("%lf", &probability);
			cin >> path;
		}
		*/
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			//cout << "asdsa" << i << endl;
			//if (i + 1 == data_samples[s].size()-1)break;
			for (int k = 0; k < NUM_OF_TREE_IN_PREV_FOREST; k++) {
				cin >> path;
				//cout << path << endl; break;
				//scanf("%lf", &.probability_estimate[k]);
				data_samples[s][i + 1].path_info_with_leaf_id[k] = binary_string_to_int(path);
				//if (s == NUM_TRAINING_SAMPLES - 1 && i == IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION - 1) cout << path << endl;
			}
			//cout << path << endl; break;
			//cout << "asdsa" <<i<< endl;

		}
		
		//cout << "data loaded" << endl;
	}
	//cout << "yyyyyyyyy" << data_samples[0][1].path_info_with_leaf_id[0] << " "<<data_samples[0][1].path_info_with_leaf_id[1]<<" "<<data_samples[0][1].path_info_with_leaf_id[2]<<endl;

	//cout << "xxxxxxxxx" << data_samples[98][IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION].path_info_with_leaf_id[9] <<" "<< data_samples[98][IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION].path_info_with_leaf_id[8]<<" "<< data_samples[98][IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION].path_info_with_leaf_id[7] <<endl;
	/*
	for (int s = 0; s < NUM_TRAINING_SAMPLES; s++) {
		scanf("%d", &classname);
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &data_samples[s][i+1].path_length);

		}
	}

	for (int s = 0; s < NUM_TRAINING_SAMPLES; s++) {
		scanf("%d", &classname);
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &data_samples[s][i+1].shift_diff);

		}
	}*/
	
	//cout << "N: " << count << endl;
	//sort(sortedSamples, sortedSamples + NUM_TRAINING_SAMPLES, comp);
	/*
	for (int k = 0; k < NUM_OF_CLASSES; k++) {
		cout << k << " : " << training_data_dist[k] << endl;
	}
	
	for (int k = 0; k < NUM_TRAINING_SAMPLES; k++) {
		cout << sortedSamples[k].classname << " " << sortedSamples[k].sample_index<<endl;
	}
	*/
}
void loadTestSamples(vector<vector<features>> &data_samples, int num_of_samples) {
	freopen("test_data_set_scaled.txt", "r", stdin);
	int classname;
	vector<features> values;
	struct features feature;
	int intensity;
	double probability;
	int index = 0;
	int count = 0;
	//cout << "gggg" << endl;
	for (int s = 0; s < num_of_samples; s++) {
		scanf("%d", &classname);
		//while (scanf("%d", &classname)==1) {
		//count++;
		//cout << "dddd" << endl;

		//cout << classname<<" ";
		feature.class_name = classname;
		values.clear();
		values.push_back(feature);
		//training_data_dist[classname]++;
		//sortedSamples[index].classname = classname;
		//sortedSamples[index].sample_index = index;
		//index++;
		//feature.class_name = classname;
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &feature.intensity);
			//feature.intensity = intensity;

			values.push_back(feature);
			//cout << intensity << " ";

		}
		//cout << endl;
		data_samples.push_back(values);
		//if(i==10)break;

		//}
	}

	for (int s = 0; s < num_of_samples; s++) {

		for (int k = 0; k < NUM_OF_CLASSES; k++) {
			scanf("%lf", &probability);
		}
		
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			for (int k = 0; k < NUM_OF_CLASSES; k++) {
				scanf("%lf", &data_samples[s][i + 1].probability_estimate[k]);
			}
		}
	}

	string path;
	for (int s = 0; s < num_of_samples; s++) {

		/*
		for (int k = 0; k < NUM_OF_TREE_IN_PREV_FOREST; k++) {
		//scanf("%lf", &probability);
		cin >> path;
		}
		*/
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			//cout << "asdsa" << i << endl;
			//if (i + 1 == data_samples[s].size()-1)break;
			for (int k = 0; k < NUM_OF_TREE_IN_PREV_FOREST; k++) {
				cin >> path;
				//cout << path << endl; break;
				//scanf("%lf", &.probability_estimate[k]);
				data_samples[s][i + 1].path_info_with_leaf_id[k] = binary_string_to_int(path);
				//if (s == NUM_TRAINING_SAMPLES - 1 && i == IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION - 1) cout << path << endl;
			}
			//cout << path << endl; break;
			//cout << "asdsa" <<i<< endl;

		}

		//cout << "data loaded" << endl;
	}

	/*
	for (int s = 0; s < num_of_samples; s++) {
		scanf("%d", &classname);
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &data_samples[s][i+1].path_length);

		}
	}

	for (int s = 0; s < num_of_samples; s++) {
		scanf("%d", &classname);
		for (int i = 0; i < IMAGE_DATA_DIMENSION*IMAGE_DATA_DIMENSION; i++) {
			scanf("%d", &data_samples[s][i+1].shift_diff);

		}
	}
	*/
}
int main() {
	
	freopen("output.txt", "w", stdout);
	printParameters();
	/*
	map<int, int> mymap;
	mymap[1] = 5;
	mymap[2] = 6;
	mymap[15] = 10;
	mymap[20]++;
	int a = 5;
	a += 6 * 5;
	cout << a << endl;
	int row;
	int col;
	for (int i = 0; i < 100000; i++) {
		row = (int)row_distribution(generator);
		col = (int)row_distribution(generator);
		//cout << row << "," << col <<endl;
		if (row < 29 && col < 29 && row >= 0 && col >= 0) {
			normal[row][col]++;
		}
	}
	for (int i = 0; i < 29; i++) {
		for (int j = 0; j < 29; j++) {
			cout <<normal[i][j];
			if (j < 28) { cout << ","; }
		}
		cout<<endl;
	}
	cout << endl;
	return 0;
	*/
	//maptest(mymap);
	//getchar();
	//return 0;
	//populateFilters(); return 0;
	//cout << binary_string_to_int("110") << " "<<binary_string_to_int("1000010")<<endl;
	
	long long start = clock();
	readTrainingDatafromText(training_data);
	//return 0;
	//readMNSITfromCSV("D:\\mnist_test.csv", training_data, 10000);//loads first 10000 samples for training
	long long stop = clock();
	cout << "Time needed to load data: " << stop - start << endl;
	//return 0;
	
	/*///////////Generating images for analysis///////////////////////////
	for (int s = 0; s < NUM_OF_TEST_SAMPLES; s++) {
		
		Mat vect = Mat::zeros(IMAGE_DATA_DIMENSION, IMAGE_DATA_DIMENSION, CV_8UC1);

		// Loop over vectors and add the data
		int counter = 1;
		cout << training_data.size() << endl;
		double max_prob;
		for (int r = 0; r < IMAGE_DATA_DIMENSION; r++) {
			for (int c = 0; c < IMAGE_DATA_DIMENSION; c++) {
				if (training_data[0][counter].probability_estimate[0] > training_data[0][counter].probability_estimate[1] &&
					training_data[0][counter].probability_estimate[0] > training_data[0][counter].probability_estimate[2]) {
					max_prob = training_data[0][counter].probability_estimate[0];
				}
				else if (training_data[0][counter].probability_estimate[1] > training_data[0][counter].probability_estimate[0] &&
					training_data[0][counter].probability_estimate[1] > training_data[0][counter].probability_estimate[2]) {
					max_prob = training_data[0][counter].probability_estimate[1];
				}
				else {
					max_prob = training_data[0][counter].probability_estimate[2];
				}
				vect.at<uchar>(r, c) = training_data[0][counter].intensity;// max_prob * 255;// training_data[10][counter].intensity;
				cout << (int)vect.at<uchar>(r, c) << " ";
				counter++;
			}
			cout << endl;
		}
		cout << counter << endl;
		namedWindow("Display window" + s, WINDOW_AUTOSIZE);// Create a window for display.
		imshow("Display window" + s, vect);
		waitKey();

	}
	
		getchar();
	*/
	populate_3_by_3_filters();//populateFilters();//generating the filters which will be used later
	cout << training_data.size() << endl;
	vector<vector<int>>t_samples_index;
	vector<vector<int>>h_histogram;
	vector<int>samples_index;
	vector<int>histogram;
	Node *root[NUM_OF_TREE];
	/*
	Building the Forest with NUM_OF_TREE
	*/
	//initializing the histogram of 10 classes of the digits
	for (int t = 0; t < NUM_OF_TREE; t++) {
			histogram.clear();
		for (int i = 0; i < NUM_OF_CLASSES; i++) {
			histogram.push_back(0);
		}
		h_histogram.push_back(histogram);
	}
	

	rand_seed = getTickCount();
	cout << "Main Func " << rand_seed << endl;
	srand(rand_seed);

	for (int t = 0; t < NUM_OF_TREE; t++) {
		//rand_seed += 123456789;
		//cout << "Selecting sample for tree " << rand_seed << endl;
		//srand(rand_seed);
		int total_uniform_samples = 0;
		int rand_range_from = 0;
		int rand_range_to = 0;
		for (int classname = 0; classname < NUM_OF_CLASSES; classname++) {
			rand_range_to = training_data_dist[classname];
			//cout << "range:" << rand_range_from << "to " << rand_range_to << endl;
			
			for (int numsamples = 0; numsamples < NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE / NUM_OF_CLASSES; numsamples++) {
				int i = rand_range_from + rand() % (rand_range_to);
				samples_index.push_back(sortedSamples[i].sample_index);
				if (sortedSamples[i].classname == classname) { h_histogram[t][classname]++; }
			}
			rand_range_from += rand_range_to;
			
		}

		/*
		while (1) {
			//samples_index.push_back(rand() % (NUM_TRAINING_SAMPLES));//randomly picking samples for building tree
			int i = rand() % (NUM_TRAINING_SAMPLES);
			int k = training_data[i][0];
			if (h_histogram[t][k] < NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE / NUM_OF_CLASSES) { 
				h_histogram[t][k]++;
				samples_index.push_back(i); 
				total_uniform_samples++;
			}
			//cout << samples_index[i] << endl;
			
			//test_data.push_back(training_data[samples_index.at(i)]);
			if (total_uniform_samples == NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE) { break; }
		}*/

		/*
		cout << "Tree " << t << ":" << endl;
		for (int ttt = 0; ttt < NUM_OF_CLASSES; ttt++) {
			cout << ttt<<" : " << h_histogram[t][ttt]<<endl;
		}
		*/
		t_samples_index.push_back(samples_index);
		samples_index.clear();
	}
	//return 0;
	for (int t = 0; t < NUM_OF_TREE; t++) {
		/*
		for (int i = 0; i < 10; i++) {
			cout << i << " " << h_histogram[t][i] << endl;
		}*/
		root[t] = new Node();
		tree_node_count = 0; tree_height = -1; largest_leaf_with_data_samples = 0;
		root[t] = BuildTree(root[t], 0, t_samples_index[t], h_histogram[t],1);
		cout << "Tree "<<t+1<<" built." <<"total nodes "<<tree_node_count<< "height "<<tree_height<<endl;
		forest_balance += (double)tree_node_count / pow(2, (double)tree_height - 1);// << endl;
		forest_balance_ += (double)tree_node_count / pow(2, (double)tree_height);
		forest_saturation += 1 - (double)largest_leaf_with_data_samples / NUM_OF_TRAINING_SAMPLE_FOR_EACH_TREE;
	}
	cout << "Forest balance " << forest_balance/NUM_OF_TREE*100 << endl<< " Forest balance_ " << forest_balance_/NUM_OF_TREE*100 <<" Forest saturation "<<forest_saturation/NUM_OF_TREE*100<<endl;;
	
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	
	/*
	Building the second three
	*/
	/*
	histogram.clear();
	samples_index.clear();
	for (int i = 0; i < NUM_OF_CLASSES; i++) {
		histogram.push_back(0);
	}

	for (int i = 0; i < NUM_TRAINING_SAMPLES / 10; i++) {
		samples_index.push_back(rand() % (NUM_TRAINING_SAMPLES));//randomly picking 1000 samples for building tree
		histogram[training_data[samples_index.at(i)][0]]++;
	}
	for (int i = 0; i < 10; i++) {
		cout << i << " " << histogram[i] << endl;
	}
	
	Node *root2 = new Node();
	root2 = BuildTree(root2, 0, samples_index, histogram);
	cout << "Tree2 built" << endl;
	*/
	//////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	Building the third three
	*/
	/*
	histogram.clear();
	samples_index.clear();
	for (int i = 0; i < NUM_OF_CLASSES; i++) {
		histogram.push_back(0);
	}

	for (int i = 0; i < NUM_TRAINING_SAMPLES / 10; i++) {
		samples_index.push_back(rand() % (NUM_TRAINING_SAMPLES));//randomly picking 1000 samples for building tree
		histogram[training_data[samples_index.at(i)][0]]++;
	}
	for (int i = 0; i < 10; i++) {
		cout << i << " " << histogram[i] << endl;
	}

	Node *root3 = new Node();
	root3 = BuildTree(root3, 0, samples_index, histogram);
	cout << "Tree3 built" << endl;
	*/
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	for (int kk = 0; kk < NUM_FEATURE_TYPE; kk++) {
		cout << feature_type_selection_counter[kk] << endl;
	}
	//////////////////////////////////Testing with sample from training data//////////////////////////////////////
	double** hist;
	hist = (double**)malloc(NUM_OF_TREE * sizeof(double*));
	double correct = 0;
	double E_rms = 0;
	//readMNSITfromCSV("D:\\mnist_test.csv", test_data, NUM_OF_TEST_SAMPLES);
	loadTestSamples(test_data, NUM_OF_TEST_SAMPLES);
	for (int i = 0; i < test_data.size(); i++) {
		for (int t = 0; t < NUM_OF_TREE; t++) {

			hist[t] = (double*)malloc(NUM_OF_CLASSES * sizeof(double));
			hist[t] = getPredictionHistogram(root[t], i);
		}


		double max = -999;
		int res_class;
		double res_prob;

		for (int i = 0; i < NUM_OF_CLASSES; i++) {
			double sum = 0;
			for (int t = 0; t < NUM_OF_TREE; t++) {
				sum += hist[t][i];
			}
			if (sum > max) {
				max = sum;
				res_class = i;
				res_prob = max / NUM_OF_TREE;
			}

		}
		arr_occur[test_data[i][0].class_name]++;
		E_rms += (1 - res_prob)*(1 - res_prob);
		cout << "predicted class: " << res_class << " " << "with probability of" << res_prob * 100 << "%" << " Actual class is: "<<test_data[i][0].class_name<<endl;
		if (res_class == test_data[i][0].class_name) {
			correct++;
			arr_corr[res_class]++;
			E_rms += (1 - res_prob)*(1 - res_prob);
		}
		else {
			E_rms += res_prob*res_prob;
		}
	}

	for (int td = 0; td < NUM_OF_CLASSES; td++) {
		cout << td<<": "<<arr_occur[td] << " " << arr_corr[td] << " " << (double)arr_corr[td] / arr_occur[td] * 100 << endl;
	}
	cout << "Correct prediction: " << correct / test_data.size() * 100 << "%" << endl;
	cout << "E_rms: " << E_rms / NUM_OF_TEST_SAMPLES * 100 << "%" << endl;
	/*
	cout << "Total unique response: " << global_response.size()<<endl;
	
	for (map<int, int>::iterator hist = global_response.begin(); hist != global_response.end(); ++hist) {
		if (hist->first > max_response) {
			max_response = hist->first;
		}
		if (hist->first < min_response) {
			min_response = hist->first;
		}
		cout << hist->first << ": " << hist->second << endl;
		
	}
	
	cout << "Max response: " << max_response << " " << "Min response: " << min_response << endl;
	*/
		//Now displaying first two images from data sample
	/*
	Mat vect = Mat::zeros(28, 28, CV_8UC1);
	
	// Loop over vectors and add the data
	int counter = 1;
	cout << training_data.size() << endl;
	
	for (int r = 0; r < 28; r++) {
		for (int c = 0; c < 28; c++) {
			vect.at<uchar>(r, c) = training_data[10][counter];
			//cout << (int)vect.at<uchar>(r, c)<<" ";
			counter++;
		}
		//cout << endl;
	}
	cout << counter << endl;
	namedWindow("Display window", 0.5);// Create a window for display.
	imshow("Display window", vect);
	//code for reading image files
	String path("./data/*jpg");
	vector<String> fn;
	vector<int>training_data_index;
	int sample_index = 0;
	glob(path, fn, true);
	//Mat img = imread(path,CV_LOAD_IMAGE_COLOR);
	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("Display window", img);
	*/

	/*
	for (size_t k = 0; k < fn.size(); k++) {
		Mat img = imread(fn[k], CV_LOAD_IMAGE_COLOR);
		if (!img.data) { continue; };
		training_data.push_back(img);
		training_data_index.push_back(sample_index);
		cout<<sample_index++;
		//training data labels also has to be loaded
		namedWindow("Display window"+k, 0.5);// Create a window for display.
		imshow("Display window"+k, training_data.at(k));                   // Show our image inside it.
		Mat grey;
		cvtColor(training_data.at(k), grey, CV_BGR2GRAY);
		Scalar intensity = (training_data.at(k).at<uchar>(50, 50));
		cout << (int)training_data.at(k).at<uchar>(50, 50) << endl;
		cout << intensity.val[0] << " "<<intensity.val[1]<< " "<<intensity.val[2]<<" "<<(int)grey.at<uchar>(50,50)<<endl;
	}
	
	

	Mat mytestimg = training_data.at(0);
	//cout<<mytestimg.at(Point(5,5));
	populateFilters();
	Node *root = new Node();
	root = BuildTree(root,1, training_data_index);
	getPrediction(root, 12);
	getPrediction(root, 10);
	//getPrediction(root, 8);
	*/
	//getchar();
	//read from MNIST

	vector<vector<double>> ar;
	//ReadMNIST(5, 784, ar);
	Mat mat_img;
	//cout << ar.size() <<" "<< ar.at(3).size() << endl;
	/*for (size_t i = 0; i < ar.size(); i++)
	{
		if (i == 2)break;
		for (size_t j = 0; j < ar.at(i).size(); j++)
		{
			mat_img.at<double>(i, j) = ar[i][j];
		}
		
	}*/
	//getchar();
	//waitKey(0);
	return 0;
}