use log::{debug, info};
use opencv::calib3d::{find_homography, RANSAC};
use opencv::core::{DMatch, KeyPoint, Mat, Point2f, Vector, DecompTypes};
use opencv::core::Scalar;
use opencv::imgproc::warp_perspective;


/// Takes to list of keypoints and tries to match each one to another.
/// This matching algorithm matches each point $P$ points from the `source` dataset,
/// to the closest point $P'$ in the `target` dataset.
///
/// # Complexity
/// The runtime complexity is $n*m$ with $n$ as the number of `KeyPoint`s in
/// `source` and $m$ the number of `KeyPoint`s in `target`.
///
/// # Arguments
/// * source
/// * target
///
/// # Returns
/// A `Vec` of `DMatch`. Each containing information about the index of `source` set (`query_idx`)
/// and the `target` set (`train_idx`) and the pixel distance between both points $P$ and $P'$.
fn min_pos_delta_match(source: &Vector<KeyPoint>, target: &Vector<KeyPoint>) -> Vec<DMatch> {
    fn find_nearest(point: &KeyPoint, point_idx: usize, target_set: &Vector<KeyPoint>) -> DMatch {

        let mut lowest_idx: usize = 0;
        let mut lowest_dist: f32 = f32::MAX;

        target_set.to_vec().iter().enumerate().for_each(|(i, p)| {
            let dist = p.pt - point.pt;
            let dist = dist.dot(dist);
            if lowest_dist > dist {
                lowest_idx = i;
                lowest_dist = dist;
            }
        });

        DMatch::new(point_idx as i32, lowest_idx as i32, lowest_dist.sqrt()).unwrap()
    }

    let mut src: Vec<i32> = vec![];
    let mut dest: Vec<i32> = vec![];
    source.iter()
        .enumerate()
        .map(|(i, point)| find_nearest(&point, i, target))
        .filter(|m| {
            if src.contains(&m.query_idx) || dest.contains(&m.train_idx) { return false }
            src.push(m.query_idx);
            dest.push(m.train_idx);
            true
        })
        .collect()
}

pub fn get_matches(
    source: &Vector<KeyPoint>,
    dest: &Vector<KeyPoint>
) -> Vector<DMatch> {
    let matches: Vector<DMatch> = Vector::from_iter(
        min_pos_delta_match(source, dest)
            .into_iter()
            .filter(|x| x.distance < 20.0),
    );
    matches.iter().for_each(|m| {
        debug!(
            "Match {} to {} with distance: {}",
            m.train_idx, m.query_idx, m.distance
        )
    });
    matches
}


/// Returns the homography of
///
/// # Arguments
/// * `matches`:
/// * `source_kp`:
/// * `dest_kp`:
///
/// # Returns
pub fn find_alignment_homography(
    matches: &Vector<DMatch>,
    source_kp: &Vector<KeyPoint>,
    dest_kp: &Vector<KeyPoint>,
) -> Mat {
    let src: Vector<Point2f> = matches
        .iter()
        .map(|mat| source_kp.get(mat.query_idx as usize).unwrap().pt)
        .collect();

    let dest: Vector<Point2f> = matches
        .iter()
        .map(|mat| dest_kp.get(mat.train_idx as usize).unwrap().pt)
        .collect();

    find_homography(&src,
                    &dest,
                    &mut Mat::default(),
                    RANSAC,
                    5.0
    ).unwrap()
}


pub fn align_series(images: &[Mat], keypoints: &[Vector<KeyPoint>]) -> Vec<Mat> {

    let homographies = images.iter().skip(1).enumerate().map(|(i, im)| {
        info!("Calculating Homography for {} to {}", i, i-1);
        let matches = get_matches(&keypoints[i], &keypoints[i - 1]);
        find_alignment_homography(&matches, &keypoints[i], &keypoints[i - 1])
    }).collect::<Vec<Mat>>();


    images.iter().skip(1).zip(homographies).map(|(im, homography)| {
        let mut warped = Mat::default();
        warp_perspective(&im,
                         &mut warped,
                         &homography,
                         opencv::core::Size_ { width: 1867, height: 2800 },
                         0,
                         0,
                         Scalar::default());
        warped
    }).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_delta_match() {
        let kp1 = Vector::from_iter(vec![
            KeyPoint::new_coords(10.0, 10.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
            KeyPoint::new_coords(0.0, 10.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
        ]);
        let kp2 = Vector::from_iter(vec![
            KeyPoint::new_coords(0.0, 9.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
            KeyPoint::new_coords(10.0, 9.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
        ]);

        let result = min_pos_delta_match(&kp1, &kp2);

        assert_eq!(result[0].train_idx, 1);
        assert_eq!(result[1].train_idx, 0);
    }
}
