def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    aa=max(0,ax2-ax1)*max(0,ay2-ay1)
    ab=max(0,bx2-bx1)*max(0,by2-by1)
    union=aa+ab-inter
    return inter/union if union else 0.0


def best_iou(pred_boxes,true_boxes):
    if not pred_boxes or not true_boxes: return 0.0
    return max(iou(p,t) for p in pred_boxes for t in true_boxes)
