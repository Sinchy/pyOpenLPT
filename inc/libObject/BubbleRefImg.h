#ifndef BUBBLEREFIMG_H
#define BUBBLEREFIMG_H

#include <vector>
#include <typeinfo>

#include "BubbleResize.h"
#include "ObjectInfo.h"
#include "Matrix.h"
#include "STBCommons.h"
#include "myMATH.h"


class BubbleRefImg {
public:
    // user needs to make sure cam_list.useid_list is well defined
    BubbleRefImg() {};

    ~BubbleRefImg() {};

    bool calBubbleRefImg(const std::vector<std::unique_ptr<Object3D>>& objs_out,
                         const std::vector<std::vector<std::unique_ptr<Object2D>>>& bb2d_list_all,
                         const std::vector<Camera>& cams,
                         const std::vector<Image>& img_input, 
                         double r_thres = 6, int n_bb_thres = 5);

    const Image& operator[](int camID) const{
        return _img_Ref_list[camID];
    };

    double getIntRef(int camID) const {
        return _intRef_list[camID];
    };

    bool _is_valid = false;

private:
    std::vector<Image> _img_Ref_list; // reference images for each camera
    std::vector<double> _intRef_list;  // average intensity of reference images for each camera
    
};


#endif // BUBBLEREFIMG_H