import os
import os.path as osp

def is_ftype(pth:os.PathLike, ftype:str=None)->bool:
    
    """
    if ftype = None, it means any type is ok !
    """
    
    if ftype is None: 
        return True
    if pth[-len(ftype):] == ftype:
        return True

def make_folder(d:os.PathLike) -> os.PathLike:
    if not osp.exists(d):
        os.mkdir(d)
    return d

def all_files(root:os.PathLike, only_want_ftype:str=None)->list:
    
    ret = []
    for r, _, f in os.walk(root):
        for fi in f:
            ret.append(osp.join(r, fi))
    ret = list(filter(lambda x:is_ftype(x, only_want_ftype), ret))
    
    return ret

def build_file_tree(refer_root:os.PathLike, to_root:os.PathLike)->None:
    """
    Recursively construct a file tree mirroring the structure of reference_root.
   
    - refer_root : reference root    
    - to_root    : the root of new building file tree. *The up dir of to_root must exist*
    """
    tr = make_folder(to_root)
    # build all folder under refer_root
    for fi in list(_ for _ in  os.listdir(refer_root) if osp.isdir(osp.join(refer_root, _))):
        _ = build_file_tree(
            refer_root=osp.join(refer_root, fi), 
            to_root=make_folder(osp.join(tr, fi))
        )  
    return to_root
if __name__ == "__main__":
    # build_file_tree(refer_root=osp.join("..", "refer"),to_root=osp.join("..", "copyrefer"))
    pass