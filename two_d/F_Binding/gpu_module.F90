
module gpu_module

  use, intrinsic :: iso_c_binding

  implicit none

  interface

    subroutine posson_c() bind(C, name = "posson_c_")
    end subroutine posson_c

  end interface

end module gpu_module


