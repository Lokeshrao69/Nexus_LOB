//
// Nexus-LOB :: Pybind11 bridge (SHARED SEAM — bindings/)
// Owner: interface architect. Wires the C++ matching engine (cpp_engine/) to
// Python: (a) zero-copy live views of the L2 ladder for the RL hot path,
// (b) owning copies (snapshot) for telemetry / tests, and (c) order-entry
// methods (limit / market / cancel / modify) that drive the real engine. See
// bindings/CONTRACT.md for the state-contract spec.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nexus/types.hpp"
#include "nexus/limit_order_book.hpp"
#include "risk_cpu.hpp"  // subsystem 3: CPU Monte-Carlo VaR/CVaR reference

namespace py = pybind11;
using nexus::BookStateView;
using nexus::ExecResult;
using nexus::Fill;
using nexus::OrderId;
using nexus::Price;
using nexus::Qty;
using nexus::Side;
using nexus::Status;
using nexus::TimeInForce;

namespace {

// One pybind-facing `Engine` owns a real matching book by value and delegates to
// it. The book's BookStateView is a *stable member* of the book, so the zero-copy
// views exposed here remain valid for as long as the Engine object is alive
// (CONTRACT.md §5 "Lifetime & concurrency").
class Engine {
public:
    // Defaults: a wide-but-memory-sane price band (1..100_000 ticks — each price
    // ticks into a pre-sized level slot) plus a large resting pool, so `Engine()`
    // (as the parity test and the env call it) just works. Callers that need a
    // different band / capacity can pass args.
    Engine(Price min_price = 1, Price max_price = 100'000,
           std::size_t pool_capacity = std::size_t(1) << 18)
        : book_(min_price, max_price, pool_capacity) {}

    // LIVE WINDOW — aliases engine memory; valid only until the next mutating call.
    const BookStateView& state() const noexcept { return book_.view(); }

    // ---- order entry: clear the previous call's fills, then delegate ----------
    ExecResult submit_limit(OrderId id, Side side, Price price, Qty qty,
                            TimeInForce tif = TimeInForce::GTC) {
        fills_.clear();
        return book_.submit_limit(id, side, price, qty, tif, &fills_);
    }
    ExecResult submit_market(OrderId id, Side side, Qty qty) {
        fills_.clear();
        return book_.submit_market(id, side, qty, &fills_);
    }
    ExecResult cancel(OrderId id) {
        fills_.clear();
        return book_.cancel(id);
    }
    ExecResult modify(OrderId id, Price new_price, Qty new_qty) {
        fills_.clear();
        return book_.modify(id, new_price, new_qty, &fills_);
    }

    // ---- introspection (delegate to the book) ---------------------------------
    Price  best_bid()        const noexcept { return book_.best_bid(); }
    Price  best_ask()        const noexcept { return book_.best_ask(); }
    bool   has_bid()         const noexcept { return book_.has_bid(); }
    bool   has_ask()         const noexcept { return book_.has_ask(); }
    Price  spread()          const noexcept { return book_.spread(); }
    std::size_t live_orders() const noexcept { return book_.live_orders(); }

    // Executions from the MOST RECENT order call (empty if nothing crossed).
    const std::vector<Fill>& last_fills() const noexcept { return fills_; }

private:
    std::vector<Fill> fills_;       // regenerated per order call
    nexus::LimitOrderBook book_;
};

// Zero-copy, 1-D array over `ptr` of length `n`, keeping `owner` (the engine)
// alive for the array's lifetime. A non-null `base` => pybind does NOT copy.
template <typename T>
py::array_t<T> view_1d(const T* ptr, std::size_t n, py::handle owner) {
    return py::array_t<T>({static_cast<py::ssize_t>(n)},
                          {static_cast<py::ssize_t>(sizeof(T))},
                          ptr, owner);
}

// Owning (copied) 1-D array — safe to retain across engine mutations.
template <typename T>
py::array_t<T> copy_1d(const T* ptr, std::size_t n) {
    auto a = py::array_t<T>(static_cast<py::ssize_t>(n));
    std::memcpy(a.mutable_data(), ptr, n * sizeof(T));
    return a;
}

template <bool ZeroCopy>
py::dict make_payload(py::object self, const BookStateView& s) {
    py::dict d;
    if constexpr (ZeroCopy) {
        d["bid_px"] = view_1d(s.bid_px, nexus::kDepth, self);
        d["bid_sz"] = view_1d(s.bid_sz, nexus::kDepth, self);
        d["bid_ct"] = view_1d(s.bid_ct, nexus::kDepth, self);
        d["ask_px"] = view_1d(s.ask_px, nexus::kDepth, self);
        d["ask_sz"] = view_1d(s.ask_sz, nexus::kDepth, self);
        d["ask_ct"] = view_1d(s.ask_ct, nexus::kDepth, self);
    } else {
        d["bid_px"] = copy_1d(s.bid_px, nexus::kDepth);
        d["bid_sz"] = copy_1d(s.bid_sz, nexus::kDepth);
        d["bid_ct"] = copy_1d(s.bid_ct, nexus::kDepth);
        d["ask_px"] = copy_1d(s.ask_px, nexus::kDepth);
        d["ask_sz"] = copy_1d(s.ask_sz, nexus::kDepth);
        d["ask_ct"] = copy_1d(s.ask_ct, nexus::kDepth);
    }
    d["seq"]             = s.seq;
    d["ts_ns"]           = s.ts_ns;
    d["cum_volume"]      = s.cum_volume;
    d["last_trade_px"]   = s.last_trade_px;
    d["last_trade_sz"]   = s.last_trade_sz;
    d["last_trade_side"] = s.last_trade_side;
    return d;
}

// {id, status, filled, resting} for an order-entry result.
py::dict result_dict(const ExecResult& r) {
    py::dict d;
    d["id"]      = r.id;
    d["status"]  = r.status;
    d["filled"]  = r.filled;
    d["resting"] = r.resting;
    return d;
}

// list[(maker_id, taker_id, px, qty, aggressor)] — the ITCH-style trade prints.
py::list fills_list(const std::vector<Fill>& fills) {
    py::list out;
    for (const Fill& f : fills)
        out.append(py::make_tuple(f.maker_id, f.taker_id, f.px, f.qty, f.aggressor));
    return out;
}

} // namespace

PYBIND11_MODULE(nexus_engine, m) {
    m.doc() = "Nexus-LOB C++ matching-engine bridge (state contract v"
              + std::to_string(nexus::kBookStateContractVersion) + ")";
    m.attr("CONTRACT_VERSION") = nexus::kBookStateContractVersion;
    m.attr("DEPTH")            = static_cast<std::size_t>(nexus::kDepth);
    // Exposed so tests/test_abi_parity.py can assert C++ sizeof == NumPy itemsize.
    m.attr("STATE_NBYTES")     = static_cast<std::size_t>(sizeof(BookStateView));

    py::enum_<Side>(m, "Side")
        .value("Bid", Side::Bid)
        .value("Ask", Side::Ask)
        .value("None_", Side::None);

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("GTC", TimeInForce::GTC)
        .value("IOC", TimeInForce::IOC)
        .value("FOK", TimeInForce::FOK);

    py::enum_<Status>(m, "Status")
        .value("Accepted", Status::Accepted)
        .value("Filled", Status::Filled)
        .value("PartiallyFilledResting", Status::PartiallyFilledResting)
        .value("Canceled", Status::Canceled)
        .value("Rejected_DupId", Status::Rejected_DupId)
        .value("Rejected_BadPrice", Status::Rejected_BadPrice)
        .value("Rejected_BadQty", Status::Rejected_BadQty)
        .value("Rejected_PoolFull", Status::Rejected_PoolFull)
        .value("Rejected_FOK", Status::Rejected_FOK)
        .value("NoOp", Status::NoOp);

    py::class_<Engine>(m, "Engine")
        .def(py::init<Price, Price, std::size_t>(),
             py::arg("min_price")     = 1,
             py::arg("max_price")     = 100'000,
             py::arg("pool_capacity") = std::size_t(1) << 18,
             "Construct a matching engine over a fixed integer-tick price band.")

        // ---- order entry ------------------------------------------------------
        .def("submit_limit",
             [](Engine& e, OrderId id, Side side, Price price, Qty qty, TimeInForce tif) {
                 return result_dict(e.submit_limit(id, side, price, qty, tif));
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"),
             py::arg("tif") = TimeInForce::GTC,
             "Submit a limit order. Returns {id, status, filled, resting}.")
        .def("submit_market",
             [](Engine& e, OrderId id, Side side, Qty qty) {
                 return result_dict(e.submit_market(id, side, qty));
             },
             py::arg("order_id"), py::arg("side"), py::arg("qty"),
             "Submit a marketable order (crosses at any price, never rests).")
        .def("cancel",
             [](Engine& e, OrderId id) {
                 return result_dict(e.cancel(id));
             },
             py::arg("order_id"),
             "Cancel a resting order by id (NoOp if it isn't live).")
        .def("modify",
             [](Engine& e, OrderId id, Price new_price, Qty new_qty) {
                 return result_dict(e.modify(id, new_price, new_qty));
             },
             py::arg("order_id"), py::arg("new_price"), py::arg("new_qty"),
             "Modify a resting order (in-place reduce keeps time priority; "
             "reprice / size-up loses it and may cross).")

        // ---- fills / introspection --------------------------------------------
        .def("fills",
             [](Engine& e) { return fills_list(e.last_fills()); },
             "Executions from the most recent order call, as "
             "(maker_id, taker_id, px, qty, aggressor) tuples.")
        .def("best_bid",    [](Engine& e) { return e.best_bid(); },
             "Best resting bid price in ticks (0 if none).")
        .def("best_ask",    [](Engine& e) { return e.best_ask(); },
             "Best resting ask price in ticks (0 if none).")
        .def("spread",      [](Engine& e) { return e.spread(); },
             "Best ask - best bid in ticks (0 if either side is empty).")
        .def("live_orders", [](Engine& e) { return e.live_orders(); },
             "Number of orders currently resting on the book.")

        // ---- ZERO-COPY LIVE VIEW (RL hot path) --------------------------------
        .def("view",
             [](py::object self) {
                 return make_payload<true>(self, self.cast<Engine&>().state());
             },
             "Zero-copy live view of the book (mind the lifetime caveat).")

        // ---- OWNING SNAPSHOT (telemetry / tests / cross-thread) ---------------
        .def("snapshot",
             [](py::object self) {
                 return make_payload<false>(self, self.cast<Engine&>().state());
             },
             "Owning copy of the book state — safe to retain.")
        ;

    // ------------------------------------------------------------------
    // Subsystem 3 — Monte-Carlo VaR/CVaR (CPU reference, subsystem 3)
    // Exposed so Python can diff it bit-for-bit against a NumPy oracle
    // (tests/test_risk_parity.py). GPU path is separate (cuda_risk/*.cu).
    // ------------------------------------------------------------------
    m.def("compute_var_cvar",
          [](double s0, double mu, double sigma, double T, int steps, int n_paths,
             double alpha, std::uint64_t seed, double lambda_jump,
             double jump_mu, double jump_sigma) {
              nexus::risk::RiskParams p;
              p.s0 = s0;  p.mu = mu;  p.sigma = sigma;  p.T = T;
              p.steps = steps;  p.n_paths = n_paths;  p.alpha = alpha;
              p.seed = seed;
              p.lambda_jump = lambda_jump;  p.jump_mu = jump_mu;
              p.jump_sigma = jump_sigma;
              const auto r = nexus::risk::compute_var_cvar_cpu(p);
              py::dict d;
              d["var"] = r.var;  d["cvar"] = r.cvar;
              d["mean_loss"] = r.mean_loss;  d["n"] = r.n;
              return d;
          },
          py::arg("s0") = 100.0, py::arg("mu") = 0.05, py::arg("sigma") = 0.25,
          py::arg("T") = 1.0, py::arg("steps") = 252, py::arg("n_paths") = 200000,
          py::arg("alpha") = 0.95, py::arg("seed") = 0x51ED,
          py::arg("lambda_jump") = 0.0, py::arg("jump_mu") = 0.0,
          py::arg("jump_sigma") = 0.0,
          "Monte-Carlo VaR/CVaR over a GBM (or jump-diffusion) process, as "
          "fractions of the initial notional. CPU reference — the GPU kernel "
          "shares its exact RNG, so results match bit-for-bit on a CUDA box.");
}
