module ms_solver_module
  implicit none

  ! DP5(4)7M Butcher tableau — Hairer, Norsett, Wanner (1993), Solving ODEs I, Table 5.2
  double precision, parameter :: a21=0.2d0, a31=3.0d0/40.0d0, a32=9.0d0/40.0d0
  double precision, parameter :: a41=44.0d0/45.0d0, a42=-56.0d0/15.0d0, a43=32.0d0/9.0d0
  double precision, parameter :: a51=19372.0d0/6561.0d0, a52=-25360.0d0/2187.0d0
  double precision, parameter :: a53=64448.0d0/6561.0d0, a54=-212.0d0/729.0d0
  double precision, parameter :: a61=9017.0d0/3168.0d0, a62=-355.0d0/33.0d0
  double precision, parameter :: a63=46732.0d0/5247.0d0, a64=49.0d0/176.0d0, a65=-5103.0d0/18656.0d0
  double precision, parameter :: b1=35.0d0/384.0d0, b2=0.0d0, b3=500.0d0/1113.0d0
  double precision, parameter :: b4=125.0d0/192.0d0, b5=-2187.0d0/6784.0d0, b6=11.0d0/84.0d0
  double precision, parameter :: c2=0.2d0, c3=0.3d0, c4=0.8d0, c5=8.0d0/9.0d0
  double precision, parameter :: d1=5179.0d0/57600.0d0, d2=0.0d0, d3=7571.0d0/16695.0d0
  double precision, parameter :: d4=393.0d0/640.0d0, d5=-92097.0d0/339200.0d0
  double precision, parameter :: d6=187.0d0/2100.0d0, d7=1.0d0/40.0d0
  ! Integration defaults
  double precision, parameter :: RTOL_DF = 1.0d-8, ATOL_DF = 1.0d-10
  integer, parameter :: MAX_STEPS_DF = 200000

contains

  subroutine find_segment_index(t, n_var, n_pts, bc_arr, idx)
    implicit none
    double precision, intent(in) :: t
    integer, intent(in) :: n_var, n_pts
    double precision, intent(in):: bc_arr(5, n_var, n_pts)
    integer, intent(out) :: idx
    integer :: low, high, mid

    low = 1
    high = n_pts
    idx = 1
    do while (low <= high)
       mid = (low + high) / 2
       if (bc_arr(1, 1, mid) <= t) then
          idx = mid
          low = mid + 1
       else
          high = mid - 1
       end if
    end do
    if (idx < 1) idx = 1
    if (idx > n_pts - 1) idx = n_pts - 1
  end subroutine find_segment_index

  subroutine find_start_idx(log_az, end_idx_fort, k_code, k_start_factor, start_idx_fort)
    ! Binary search over monotonic log(aH) = n_bg + log(z_bg) array.
    implicit none
    integer, intent(in) :: end_idx_fort
    double precision, intent(in):: log_az(end_idx_fort)
    double precision, intent(in) :: k_code, k_start_factor
    integer, intent(out) :: start_idx_fort
    integer :: low, high, mid
    double precision :: target

    target = log(k_code) - log(k_start_factor)
    low = 1
    high = end_idx_fort
    start_idx_fort = 1
    do while (low <= high)
       mid = (low + high) / 2
       if (log_az(mid) <= target) then
          start_idx_fort = mid
          low = mid + 1
       else
          high = mid - 1
       end if
    end do
    ! Check if the next element is closer to target (closest match, not lower bound)
    if (start_idx_fort < end_idx_fort) then
       if (abs(log_az(start_idx_fort + 1) - target) < abs(log_az(start_idx_fort) - target)) then
          start_idx_fort = start_idx_fort + 1
       end if
    end if
    start_idx_fort = min(max(start_idx_fort, 1), end_idx_fort)
  end subroutine find_start_idx

  ! ── rhs_eval: MS ODE right-hand side ─────────────────────────────────
  ! bc_arr layout: (5, n_var, n_pts) with Fortran column-major ordering.
  ! bc_arr(1, var, idx) = time grid   bc_arr(2, var, idx) = a coeff
  ! bc_arr(3, var, idx) = b coeff     bc_arr(4, var, idx) = c coeff
  ! bc_arr(5, var, idx) = d coeff
  ! Spline evaluation is inlined with a single dt (all variables share
  ! the same time grid, so dt = t - t_grid(idx) is computed once).
  subroutine rhs_eval(t, vars_8, n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, alpha, &
                      use_spline, idx, deriv_vars)
    implicit none
    double precision, intent(in) :: t
    double precision, intent(in) :: vars_8(8)
    integer, intent(in) :: n_var, n_pts
    double precision, intent(in):: bc_arr(5, n_var, n_pts)
    double precision, intent(in) :: k_rel, ni, Si2, v0, alpha
    integer, intent(in) :: use_spline
    integer, intent(inout) :: idx
    double precision, intent(out) :: deriv_vars(8)

    double precision :: dt, x, y, z, n_rel, df_val, d2f_val, v0_dfdx, k2a2
    double precision :: v, vt, u, ut, h, ht, g, gt
    double precision :: exp_neg_alpha_x, dydt, m2, z_safe

    ! Segment lookup — cached idx, only re-search on boundary miss
    if (t < bc_arr(1, 1, idx) .or. t > bc_arr(1, 1, idx + 1)) then
       call find_segment_index(t, n_var, n_pts, bc_arr, idx)
    end if

    ! Inline Horner spline evaluation (all vars share the same t-grid)
    dt = t - bc_arr(1, 1, idx)
    x  = bc_arr(2, 1, idx) + dt*(bc_arr(3, 1, idx) + dt*(bc_arr(4, 1, idx) + dt*bc_arr(5, 1, idx)))
    y  = bc_arr(2, 2, idx) + dt*(bc_arr(3, 2, idx) + dt*(bc_arr(4, 2, idx) + dt*bc_arr(5, 2, idx)))
    z  = bc_arr(2, 3, idx) + dt*(bc_arr(3, 3, idx) + dt*(bc_arr(4, 3, idx) + dt*bc_arr(5, 3, idx)))
    n_rel = bc_arr(2, 4, idx) + dt*(bc_arr(3, 4, idx) + dt*(bc_arr(4, 4, idx) + dt*bc_arr(5, 4, idx)))
    n_rel = n_rel - ni

    if (use_spline == 1) then
       df_val  = bc_arr(2, 6, idx) + dt*(bc_arr(3, 6, idx) + dt*(bc_arr(4, 6, idx) + dt*bc_arr(5, 6, idx)))
       d2f_val = bc_arr(2, 7, idx) + dt*(bc_arr(3, 7, idx) + dt*(bc_arr(4, 7, idx) + dt*bc_arr(5, 7, idx)))
    else
       exp_neg_alpha_x = exp(-alpha * x)
       df_val = 2.0d0 * alpha * exp_neg_alpha_x * (1.0d0 - exp_neg_alpha_x)
       d2f_val = 2.0d0 * alpha**2 * exp_neg_alpha_x * (2.0d0 * exp_neg_alpha_x - 1.0d0)
    end if

    v0_dfdx = v0 * df_val * Si2
    dydt = -3.0d0 * z * y - v0_dfdx
    k2a2 = k_rel**2 * exp(-2.0d0 * n_rel)

    ! Guard against near-zero z (Hubble rate vanishing at end of inflation)
    z_safe = max(abs(z), 1.0d-30)

    m2 = 2.5d0 * y**2 + 2.0d0 * y * dydt / z_safe + 2.0d0 * z**2 &
         + 0.5d0 * y**4 / (z_safe**2) - v0 * d2f_val * Si2 - k2a2

    v  = vars_8(1); vt = vars_8(2)
    u  = vars_8(3); ut = vars_8(4)
    h  = vars_8(5); ht = vars_8(6)
    g  = vars_8(7); gt = vars_8(8)

    deriv_vars(1) = vt
    deriv_vars(2) = -z * vt + v * m2
    deriv_vars(3) = ut
    deriv_vars(4) = -z * ut + u * m2
    deriv_vars(5) = ht
    deriv_vars(6) = -z * ht - h * (k2a2 - 2.0d0 * z**2 + 0.5d0 * y**2)
    deriv_vars(7) = gt
    deriv_vars(8) = -z * gt - g * (k2a2 - 2.0d0 * z**2 + 0.5d0 * y**2)
  end subroutine rhs_eval

  subroutine integrate_dp5(y0, T_start, T_end, n_out, output_t, n_var, n_pts, bc_arr, &
                            k_rel, ni, Si2, v0, alpha, use_spline, out)
    implicit none
    double precision, intent(in) :: y0(8)
    double precision, intent(in) :: T_start, T_end
    integer, intent(in) :: n_out
    double precision, intent(in):: output_t(n_out)
    integer, intent(in) :: n_var, n_pts
    double precision, intent(in):: bc_arr(5, n_var, n_pts)
    double precision, intent(in) :: k_rel, ni, Si2, v0, alpha
    integer, intent(in) :: use_spline
    double precision, intent(out) :: out(8, n_out)

    double precision :: y(8), t, h, err_prev, err
    double precision :: k1(8), k2(8), k3(8), k4(8), k5(8), k6(8), y_new(8), y4(8), f0(8)
    double precision :: ym(8), sc(8), yp(8), tp, theta
    double precision :: fac, inv_err
    integer :: step, j, oi, current_idx

    y = y0
    t = T_start
    h = min(1.0d-2, (T_end - T_start) / 10.0d0)
    step = 0
    err_prev = 0.0d0
    oi = 1
    current_idx = 1

    call rhs_eval(t, y, n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, alpha, &
                  use_spline, current_idx, f0)

    do while (t < T_end .and. step < MAX_STEPS_DF .and. oi <= n_out)
       if (t + h > T_end) h = T_end - t

       k1 = f0
       call rhs_eval(t + h*c2, y + h*a21*k1, n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, &
                     alpha, use_spline, current_idx, k2)
       call rhs_eval(t + h*c3, y + h*(a31*k1 + a32*k2), n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, &
                     alpha, use_spline, current_idx, k3)
       call rhs_eval(t + h*c4, y + h*(a41*k1 + a42*k2 + a43*k3), n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, &
                     alpha, use_spline, current_idx, k4)
       call rhs_eval(t + h*c5, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4), n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, &
                     alpha, use_spline, current_idx, k5)
       call rhs_eval(t + h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5), n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, &
                     alpha, use_spline, current_idx, k6)

       y_new = y + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
       call rhs_eval(t + h, y_new, n_var, n_pts, bc_arr, k_rel, ni, Si2, v0, alpha, &
                     use_spline, current_idx, f0)
       y4 = y + h*(d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6 + d7*f0)

       ! NaN early-exit (IEEE reflexivity: NaN /= NaN)
       if (y_new(1) /= y_new(1) .or. y4(1) /= y4(1)) then
          do while (oi <= n_out)
             out(:, oi) = y
             oi = oi + 1
          end do
          return
       end if

       do j = 1, 8
          ym(j) = max(abs(y_new(j)), abs(y(j)))
          sc(j) = ATOL_DF + RTOL_DF * ym(j)
       end do

       err = 0.0d0
       do j = 1, 8
          err = err + ((y_new(j) - y4(j)) / sc(j))**2
       end do
       err = sqrt(err / 8.0d0)

       if (err <= 1.0d0) then
          yp = y; tp = t
          y = y_new; t = t + h

          do while (oi <= n_out .and. output_t(oi) <= t)
             theta = (output_t(oi) - tp) / h
             out(:, oi) = yp + theta * (y - yp)
             oi = oi + 1
          end do

          if (err > 0.0d0) then
             if (err_prev > 0.0d0) then
                fac = ((1.0d0 / err)**0.14d0) * ((err_prev / err)**0.08d0)
             else
                fac = (1.0d0 / err)**0.2d0
             end if
             h = h * min(5.0d0, max(0.1d0, 1.0d0 * fac))
          end if
          err_prev = err
       else
          if (err > 0.0d0) then
             ! Use sqrt(sqrt()) instead of **0.25 — saves 1 log + 1 exp
             inv_err = 1.0d0 / err
             h = h * max(0.1d0, 0.8d0 * sqrt(sqrt(inv_err)))
          end if
       end if

       h = max(h, 1.0d-8)
       step = step + 1
    end do

    do while (oi <= n_out)
       out(:, oi) = y
       oi = oi + 1
    end do
  end subroutine integrate_dp5

end module ms_solver_module

! ── f2py wrapper subroutines ──────────────────────────────────────────

subroutine integrate_dp5_wrapper(y0, T_start, T_end, n_out, output_t, n_var, n_pts, bc_arr, &
                                  k_rel, ni, S, v0, alpha, use_spline, out)
  use ms_solver_module
  implicit none
  double precision, intent(in) :: y0(8)
  double precision, intent(in) :: T_start, T_end
  integer, intent(in) :: n_out
  double precision, intent(in) :: output_t(n_out)
  integer, intent(in) :: n_var, n_pts
  double precision, intent(in) :: bc_arr(5, n_var, n_pts)
  double precision, intent(in) :: k_rel, ni, S, v0, alpha
  integer, intent(in) :: use_spline
  double precision, intent(out) :: out(8, n_out)

  !f2py intent(in) y0, T_start, T_end, n_out, output_t, n_var, n_pts, bc_arr, k_rel, ni, S, v0, alpha, use_spline
  !f2py intent(out) out

  double precision :: Si2
  Si2 = 1.0d0 / (S * S)

  call integrate_dp5(y0, T_start, T_end, n_out, output_t, n_var, n_pts, bc_arr, &
                     k_rel, ni, Si2, v0, alpha, use_spline, out)
end subroutine integrate_dp5_wrapper

subroutine solve_ms_grid(n_modes, k_codes, n_var, n_pts, bc_arr, &
                          n_bg_points, x_bg, y_bg, z_bg, n_bg, T_span_bg, &
                          end_idx_py, k_start_factor, S, v0, alpha, use_spline, &
                          P_S_out, P_T_out, start_idx_out)
  use ms_solver_module
  implicit none
  integer, intent(in) :: n_modes
  double precision, intent(in) :: k_codes(n_modes)
  integer, intent(in) :: n_var, n_pts
  double precision, intent(in) :: bc_arr(5, n_var, n_pts)
  integer, intent(in) :: n_bg_points
  double precision, intent(in) :: x_bg(n_bg_points), y_bg(n_bg_points), z_bg(n_bg_points), n_bg(n_bg_points)
  double precision, intent(in) :: T_span_bg(n_bg_points)
  integer, intent(in) :: end_idx_py
  double precision, intent(in) :: k_start_factor
  double precision, intent(in) :: S, v0, alpha
  integer, intent(in) :: use_spline

  double precision, intent(out) :: P_S_out(n_modes)
  double precision, intent(out) :: P_T_out(n_modes)
  integer, intent(out) :: start_idx_out(n_modes)

  !f2py intent(in) n_modes, k_codes, n_var, n_pts, bc_arr, n_bg_points, x_bg, y_bg, z_bg, n_bg, T_span_bg, end_idx_py, k_start_factor, S, v0, alpha, use_spline
  !f2py intent(out) P_S_out, P_T_out, start_idx_out

  integer :: i, start_idx_fort, end_idx_fort, jj
  double precision :: k_code, k_rel, ni, t_start, t_end, zi, vi, yv
  double precision :: y0(8), y_final(8), t_end_arr(1), out_tmp(8, 1)
  double precision :: epsH, inv_A2, zeta2, h2, pi, Si2
  double precision :: y_end, z_end, n_end_rel
  double precision :: sqrt_2k, inv_sqrt_2k, k_rel_3, k_rel_sq
  double precision, allocatable :: log_az(:)

  pi = 4.0d0 * atan(1.0d0)
  end_idx_fort = end_idx_py + 1

  ! Precompute scale-invariant quantities
  Si2 = 1.0d0 / (S * S)

  allocate(log_az(n_bg_points))
  do jj = 1, n_bg_points
     log_az(jj) = n_bg(jj) + log(max(z_bg(jj), 1.0d-300))
  end do

  ! Hoist epsH: same y_end, z_end for all modes
  y_end = y_bg(end_idx_fort)
  z_end = z_bg(end_idx_fort)
  epsH = max(y_end**2 / (2.0d0 * z_end**2), 1.0d-30)

  !$OMP PARALLEL DO &
  !$OMP DEFAULT(none) &
  !$OMP SHARED(n_modes, k_codes, n_var, n_pts, bc_arr, n_bg_points, x_bg, y_bg, z_bg, n_bg, T_span_bg) &
  !$OMP SHARED(end_idx_fort, k_start_factor, S, Si2, v0, alpha, use_spline, pi, log_az) &
  !$OMP SHARED(P_S_out, P_T_out, start_idx_out) &
  !$OMP SHARED(y_end, z_end, epsH) &
  !$OMP PRIVATE(i, start_idx_fort, k_code, k_rel, ni, t_start, t_end, zi, vi, yv) &
  !$OMP PRIVATE(y0, y_final, t_end_arr, out_tmp) &
  !$OMP PRIVATE(inv_A2, zeta2, h2, n_end_rel) &
  !$OMP PRIVATE(sqrt_2k, inv_sqrt_2k, k_rel_3, k_rel_sq) &
  !$OMP SCHEDULE(dynamic, 1)
  do i = 1, n_modes
     k_code = k_codes(i)
     call find_start_idx(log_az, end_idx_fort, k_code, k_start_factor, start_idx_fort)
     start_idx_out(i) = start_idx_fort - 1

     ni = n_bg(start_idx_fort)
     k_rel = k_code * exp(-ni)
     t_start = T_span_bg(start_idx_fort)
     t_end = T_span_bg(end_idx_fort)

     ! Precompute quantities derived from k_rel (used multiple times below)
     sqrt_2k = sqrt(2.0d0 * k_rel)
     inv_sqrt_2k = 1.0d0 / sqrt_2k
     k_rel_3 = k_rel * k_rel * k_rel
     k_rel_sq = k_rel * k_rel

     zi = z_bg(start_idx_fort)
     yv = zi / k_rel
     vi = inv_sqrt_2k

     ! Bunch-Davies initial conditions (scalar + tensor)
     y0(1) = vi
     y0(2) = k_rel * inv_sqrt_2k * yv
     y0(3) = yv * vi
     y0(4) = -k_rel * inv_sqrt_2k * (1.0d0 - yv*yv)
     y0(5) = vi
     y0(6) = k_rel * inv_sqrt_2k * yv
     y0(7) = yv * vi
     y0(8) = -k_rel * inv_sqrt_2k * (1.0d0 - yv*yv)

     t_end_arr(1) = t_end
     call integrate_dp5(y0, t_start, t_end, 1, t_end_arr, n_var, n_pts, bc_arr, &
                        k_rel, ni, Si2, v0, alpha, use_spline, out_tmp)
     y_final = out_tmp(:, 1)

     n_end_rel = n_bg(end_idx_fort) - ni
     inv_A2 = exp(-2.0d0 * n_end_rel)
     zeta2 = (y_final(1)**2 + y_final(3)**2) * inv_A2 * (S**2) / (2.0d0 * epsH)
     P_S_out(i) = (k_rel_3 * zeta2) / (2.0d0 * pi**2)

     h2 = (y_final(5)**2 + y_final(7)**2) * inv_A2 * (S**2)
     P_T_out(i) = 4.0d0 * (k_rel_3 * h2) / (pi**2)
  end do
  !$OMP END PARALLEL DO

  deallocate(log_az)

end subroutine solve_ms_grid
