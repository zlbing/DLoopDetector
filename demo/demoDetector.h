/**
 * File: demoDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#ifndef __DEMO_DETECTOR__
#define __DEMO_DETECTOR__

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include "DLoopDetector.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
struct PoseTime {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> yaw;
  std::vector<double> ts;
  std::vector<string> image_id;
};                                                                                           

/// @param TVocabulary vocabulary class (e.g: BriefVocabulary)
/// @param TDetector detector class (e.g: BriefLoopDetector)
/// @param TDescriptor descriptor class (e.g: bitset for Brief)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class demoDetector
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  demoDetector(const std::string &vocfile, const std::string &imagedir,
    const std::string &posefile, int width, int height, bool show);

  ~demoDetector() { m_metrics_write.close(); }

  /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
  void run(const std::string &name, 
    const FeatureExtractor<TDescriptor> &extractor);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
 void readPoseFile(const char *filename, std::vector<double> &xs, std::vector<double> &ys, std::vector<double> &yaws,
                   std::vector<double> &ts_, std::vector<string> &image_id) const;

 vector<string> readImageFile(const PoseTime &m_all_poses, const string &image_path);

 void writeMetrics(const DetectionResult& result, const int index);

protected:

  std::string m_vocfile;
  std::string m_imagedir;
  std::string m_posefile;
  int m_width;
  int m_height;
  bool m_show;
  PoseTime m_all_poses;
  std::map<EntryId, int> m_image_index_remap;
  ofstream m_metrics_write;
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
demoDetector<TVocabulary, TDetector, TDescriptor>::demoDetector
  (const std::string &vocfile, const std::string &imagedir,
  const std::string &posefile, int width, int height, bool show)
  : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
    m_width(width), m_height(height), m_show(show)
{
  std::string filename = m_imagedir + "/../DLoopDetector/Metrics_LoopMetrics";
  std::cout<<"Write Metrics to "<<filename<<std::endl;
  m_metrics_write.open(filename, ios::trunc);
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::run
  (const std::string &name, const FeatureExtractor<TDescriptor> &extractor)
{
  cout << "DLoopDetector Demo" << endl 
    << "Dorian Galvez-Lopez" << endl
    << "http://doriangalvez.com" << endl << endl;
  
  // Set loop detector parameters
  typename TDetector::Parameters params(m_height, m_width);
  
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  
  // We are going to change these values individually:
  params.use_nss = true; // use normalized similarity score instead of raw score
  params.alpha = 0.3; // nss threshold
  params.k = 1; // a loop must be consistent with 1 previous matches
  params.geom_check = GEOM_DI; // use direct index for geometrical checking
  params.di_levels = 2; // use two direct index levels
  
  // To verify loops you can select one of the next geometrical checkings:
  // GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
  //    the features between the two images.
  // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
  //    which makes them faster. However, creating the flann structure may
  //    be slow.
  // GEOM_DI: the direct index is used to select correspondence points between
  //    those features whose vocabulary node at a certain level is the same.
  //    The level at which the comparison is done is set by the parameter
  //    di_levels:
  //      di_levels = 0 -> features must belong to the same leaf (word).
  //         This is the fastest configuration and the most restrictive one.
  //      di_levels = l (l < L) -> node at level l starting from the leaves.
  //         The higher l, the slower the geometrical checking, but higher
  //         recall as well.
  //         Here, L stands for the depth levels of the vocabulary tree.
  //      di_levels = L -> the same as the exhaustive technique.
  // GEOM_NONE: no geometrical checking is done.
  //
  // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4 
  // yields the best results in recall/time.
  // Check the T-RO paper for more information.
  //
  
  // Load the vocabulary to use
  cout << "Loading " << name << " vocabulary..." << endl;
  TVocabulary voc(m_vocfile);
  
  // Initiate loop detector with the vocabulary 
  cout << "Processing sequence..." << endl;
  TDetector detector(voc, params);
  
  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;

  // // load image filenames  
  // vector<string> filenames = 
  //   DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true);
  // std::cout<<"load image file size="<<filenames.size()<<std::endl;
  // load robot poses
  readPoseFile(m_posefile.c_str(), m_all_poses.x, m_all_poses.y, m_all_poses.yaw, m_all_poses.ts, m_all_poses.image_id);
  std::cout<<"load pose size="<<m_all_poses.x.size()<<std::endl;
  // load image filenames
  vector<string> filenames = readImageFile(m_all_poses, m_imagedir);
  std::cout << "load image file size=" << filenames.size() << std::endl;
  // we can allocate memory for the expected number of images
  detector.allocate(filenames.size());
  
  // prepare visualization windows
  DUtilsCV::GUI::tWinHandler win = "Current image";
  DUtilsCV::GUI::tWinHandler winplot = "Trajectory";
  
  DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness
  DUtilsCV::Drawing::Plot::Style loop_style('r', 2); // color, thickness
  
  DUtilsCV::Drawing::Plot implot(240, 320,
    - *std::max_element(m_all_poses.x.begin(), m_all_poses.x.end()),
    - *std::min_element(m_all_poses.x.begin(), m_all_poses.x.end()),
    *std::min_element(m_all_poses.y.begin(), m_all_poses.y.end()),
    *std::max_element(m_all_poses.y.begin(), m_all_poses.y.end()), 20);
  
  // prepare profiler to measure times
  DUtils::Profiler profiler;
  
  int count = 0;
  // go
  for(unsigned int i = 0; i < filenames.size(); ++i)
  {
    cout << "Adding image " << i << ": " << filenames[i] << "... " << endl;
    
    // get image
    cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale
    
    // show image
    if(m_show)
      DUtilsCV::GUI::showImage(im, true, &win, 10);
    
    // get features
    profiler.profile("features");
    extractor(im, keys, descriptors);
    profiler.stop();
        
    // add image to the collection and check if there is some loop
    DetectionResult result;
    
    profiler.profile("detection");
    detector.detectLoop(keys, descriptors, result);
    m_image_index_remap[result.query] = i;
    profiler.stop();
    writeMetrics(result, i);
    if(result.detection())
    {
      cout << "- Loop found with image " << result.match << "!"
        << endl;
      ++count;
    }
    else
    {
      cout << "- No loop: ";
      switch(result.status)
      {
        case CLOSE_MATCHES_ONLY:
          cout << "All the images in the database are very recent" << endl;
          break;
          
        case NO_DB_RESULTS:
          cout << "There are no matches against the database (few features in"
            " the image?)" << endl;
          break;
          
        case LOW_NSS_FACTOR:
          cout << "Little overlap between this image and the previous one"
            << endl;
          break;
            
        case LOW_SCORES:
          cout << "No match reaches the score threshold (alpha: " <<
            params.alpha << ")" << endl;
          break;
          
        case NO_GROUPS:
          cout << "Not enough close matches to create groups. "
            << "Best candidate: " << result.match << endl;
          break;
          
        case NO_TEMPORAL_CONSISTENCY:
          cout << "No temporal consistency (k: " << params.k << "). "
            << "Best candidate: " << result.match << endl;
          break;
          
        case NO_GEOMETRICAL_CONSISTENCY:
          cout << "No geometrical consistency. Best candidate: " 
            << result.match << endl;
          break;
          
        default:
          break;
      }
    }
    
    cout << endl;
    
    // show trajectory
    if(m_show && i > 0)
    {
      if(result.detection())
        implot.line(-m_all_poses.x[i - 1], m_all_poses.y[i - 1], -m_all_poses.x[i], m_all_poses.y[i], loop_style);
      else
        implot.line(-m_all_poses.x[i - 1], m_all_poses.y[i - 1], -m_all_poses.x[i], m_all_poses.y[i], normal_style);

      DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10); 
    }
  }
  
  if(count == 0)
  {
    cout << "No loops found in this image sequence" << endl;
  }
  else
  {
    cout << count << " loops found in this image sequence!" << endl;
  } 

  cout << endl << "Execution time:" << endl
    << " - Feature computation: " << profiler.getMeanTime("features") * 1e3
    << " ms/image" << endl
    << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3
    << " ms/image" << endl;

  if(m_show) {
    cout << endl << "Press a key to finish..." << endl;
    DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 0);
  }
}

// ---------------------------------------------------------------------------

template <class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile(const char *filename, std::vector<double> &xs,
                                                                     std::vector<double> &ys, std::vector<double> &yaws,
                                                                     std::vector<double> &ts_,
                                                                     std::vector<string> &image_id) const {
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in);
  
  string s;
  double ts, x, y, t;

  auto split_string = [](const std::string &s, std::vector<std::string> &v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
      v.push_back(s.substr(pos1, pos2 - pos1));

      pos1 = pos2 + c.size();
      pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) v.push_back(s.substr(pos1));
  };

  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      if(s[0] == '#') {
        continue;
      }
      std::vector<string>string_vec;
      //timestamp,sensor_id,image_id,x,y,z,qw,qx,qy,qz
      split_string(s, string_vec, ",");
      // sscanf(s.c_str(), "%lf, %lf, %lf, %lf", &ts, &x, &y, &t);
      xs.push_back(atof(string_vec[3].c_str()));
      ys.push_back(atof(string_vec[4].c_str()));
      ts_.push_back(atof(string_vec[0].c_str()));
      yaws.push_back(atof(string_vec[6].c_str()));
      image_id.push_back(string_vec[2].c_str());
    }
  }
  
  f.close();
}
template <class TVocabulary, class TDetector, class TDescriptor>
vector<string> demoDetector<TVocabulary, TDetector, TDescriptor>::readImageFile(const PoseTime &m_all_poses,
                                                                                const string &image_path) {
  vector<string> filenames;
  for (int i = 0; i < m_all_poses.image_id.size(); i++) {
    filenames.push_back(image_path + "/" + m_all_poses.image_id[i]+".png");
  }
  return filenames;
}

template <class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::writeMetrics(const DetectionResult &result, const int index) {
    auto PoseToString = [](const double x, const double y, const double w, std::string &result) {
    result.append("{ t: [");
    result.append(std::to_string(x));
    result.append(", ");
    result.append(std::to_string(y));
    result.append(", ");
    result.append(std::to_string(0));
    result.append("], q: [");
    result.append(std::to_string(w));
    result.append(", ");
    result.append(std::to_string(0));
    result.append(", ");
    result.append(std::to_string(0));
    result.append(", ");
    result.append(std::to_string(sin(acos(w))));
    result.append("] }");
  };

  auto ImageIdToString = [](const string &image_id, std::string &result) {
    std::vector<std::string> str_vec;
    {
      std::string c = "_";
      std::string::size_type pos1, pos2;
      pos2 = image_id.find(c);
      pos1 = 0;
      while (std::string::npos != pos2) {
        str_vec.push_back(image_id.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = image_id.find(c, pos1);
      }
      if (pos1 != image_id.length()) str_vec.push_back(image_id.substr(pos1));
    }
    result.append("{ \"trajectory_id\" : ");
    result.append(str_vec[0]);
    result.append(" , \"image_index\" : ");
    result.append(str_vec[1]);
    result.append("}");
  };

  std::string out;
  out.append("{\"LoopMetrics\" : ");


  out.append(" { ");
  out.append(" \"time\" : ");
  out.append(std::to_string(m_all_poses.ts[index]));
  out.append(" , \"looped_pose\" : ");
  if (result.detection()) {
    int match_id = m_image_index_remap[result.match];
    std::string pose_str;
    PoseToString(m_all_poses.x[match_id], m_all_poses.y[match_id], m_all_poses.yaw[match_id], pose_str);
    out.append(pose_str);
  } else {
    std::string pose_str;
    PoseToString(0.0, 0.0, 1.0, pose_str);
    out.append(pose_str);
  }

  out.append(" , \"confidence\" : ");
  out.append(std::to_string(1));

  out.append(" , \"current_pose\" : ");
  int cur_id = m_image_index_remap[result.query];
  std::string pose_str;
  PoseToString(m_all_poses.x[cur_id], m_all_poses.y[cur_id], m_all_poses.yaw[cur_id], pose_str);
  out.append(pose_str);

  out.append(" , \"image_id\" : ");
  std::string image_str;
  ImageIdToString(m_all_poses.image_id[cur_id], image_str);
  out.append(image_str);

  out.append(" , \"looped_imaged_id\" : ");
  if(result.detection()) {
    int match_id = m_image_index_remap[result.match];
    std::string image_str;
    ImageIdToString(m_all_poses.image_id[match_id], image_str);
    out.append(image_str);
  } else {
    std::string image_str;
    ImageIdToString("-1_-1", image_str);
    out.append(image_str);
  }
  out.append(" } ");

  out.append(" }\n");

  m_metrics_write << out;
}

// ---------------------------------------------------------------------------

#endif

