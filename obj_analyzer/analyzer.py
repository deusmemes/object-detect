import cv2

from obj_analyzer.model import get_model_by_name
import obj_analyzer.keypoints as kp
import obj_analyzer.image as iu


def match_images(img1, img2, mask1, mask2):
    img1 = iu.convert_to_gray(img1)
    img2 = iu.convert_to_gray(img2)
    key_points_img, corners = kp.draw_key_points(img1, img2)

    perspective = iu.perspective_transform(img1, key_points_img, corners)
    mask_corners = corners.copy()
    mask_corners[:, 0] -= 255
    perspective_mask = iu.perspective_transform(mask1, mask2, mask_corners)

    return perspective, perspective_mask


def analyze(model_name, images):
    # sgm = Unet(model_name)
    # sgm.get_model()

    sgm = get_model_by_name(model_name)

    images = [iu.resize(img, (256, 256)) for img in images]
    transform_images = [images[0]]
    transform_masks = [sgm.predict(images[0])]

    for i, img in enumerate(images):
        if i == len(images) - 1:
            break
        key_points_img, corners = kp.draw_key_points(images[i+1], images[0])
        # cv2.imshow('ds', key_points_img)
        perspective = iu.perspective_transform(images[i+1], key_points_img, corners)[:, 256:]
        pred = sgm.predict(perspective)
        # cv2.imshow('ddawf', perspective)
        transform_images.append(perspective)
        transform_masks.append(pred)
        # cv2.imshow('ddawfaw', pred)
        # cv2.imwrite(f'img{i}.jpg', perspective)

    # masks = [sgm.predict(img) for img in images]

    areas = []
    imgs_with_masks = []

    for i, (img, mask) in enumerate(zip(transform_images, transform_masks)):
        img_with_mask = iu.add_mask(img, mask)
        imgs_with_masks.append(img_with_mask)
        # transform, transform_mask = match_images(transform_images[0], img, transform_masks[0], mask)
        area1 = iu.get_mask_area(transform_masks[0])
        area2 = iu.get_mask_area(mask)
        if area1 == 0:
            compare_areas = 0
        else:
            compare_areas = area2 / area1
            # compare_areas += (1 - compare_areas)
            if abs(compare_areas - 1) < 0.1:
                compare_areas = 1
        print(f"area transform {i} -> {i + 1}: {compare_areas}")
        areas.append(compare_areas)

    # for i, (img, mask) in enumerate(zip(images, masks)):
    #     img_with_mask = iu.add_mask(img, mask)
    #     imgs_with_masks.append(img_with_mask)
    #     if i == (len(images) - 1):
    #         break
    #     transform, transform_mask = match_images(images[0], images[i+1], masks[0], masks[i+1])
    #     area1 = iu.get_mask_area(transform_mask)
    #     area2 = iu.get_mask_area(masks[i+1])
    #     if area1 == 0:
    #         compare_areas = 0
    #     else:
    #         compare_areas = area2 / area1
    #         # compare_areas += (1 - compare_areas)
    #         if abs(compare_areas - 1) < 0.1:
    #             compare_areas = 1
    #     print(f"area transform {i} -> {i + 1}: {compare_areas}")
    #     areas.append(compare_areas)

    return imgs_with_masks, areas
