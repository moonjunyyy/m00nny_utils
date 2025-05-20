#ifndef __TASK_H__
#define __TASK_H__

#include <any>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <variant> 
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "utils.h"
#include "random_generator.h"

namespace utils::task {

class TaskException : public std::runtime_error {
public: 
    explicit TaskException(const std::string& msg): std::runtime_error(msg) {}
};

namespace TaskUtils {
    inline std::string generateRandomHex(size_t length) { return utils::random::getGlobalRandomGenerator().hex<std::string>(length); }
    inline std::string generateUUID() { return utils::random::getGlobalRandomGenerator().uuid<std::string>(); }
}

// Forward declarations
class _Worker;
class WorkerPool;
class _TaskInterface;
template<typename _T> class _TaskBase;
template<typename Result, typename... Args> class Task;
struct _TaskPtrComparator {
    bool operator()(const std::shared_ptr<_TaskInterface>& lhs, const std::shared_ptr<_TaskInterface>& rhs) const { return (*lhs) < (*rhs); }
};

using _TaskPtr = std::shared_ptr<_TaskInterface>;
using _TaskID  = std::string;

inline static thread_local _Worker* current_worker_ = nullptr;
inline static std::map<_TaskID, _TaskPtr> all_tasks_;
inline static std::mutex all_tasks_mutex_;

class _Worker {
public:
    
    _Worker(WorkerPool* pool_ptr) : pool_(pool_ptr){
        thread_ = std::thread(worker_loop, this);
    }
    ~_Worker() { if (thread_.joinable()) thread_.join(); }
    _Worker  (const _Worker&) = delete;
    _Worker  (_Worker&&)      = delete;
    _Worker& operator         = (const _Worker&) = delete;
    _Worker& operator         = (_Worker&&)      = delete;
    
    bool try_poll_once() {
        auto task = pool_->tryGetTask();
        if (!task) return false;
        task->execute();
        return true;
    }

private: 
    WorkerPool* pool_;
    std::thread thread_;
    void worker_loop() {
        current_worker_ = this;
        while (!pool_->is_shutting_down()) {
            auto task = pool_->getTaskBlocking(); // Use the blocking version
            if (!task) {
                if (pool_->is_shutting_down()) break; // Exit if shutting down
                // else continue if woken up spuriously or queue became empty after wake-up but before pop
                continue;
            }
            task->execute();
        }
    }
};

class WorkerPool {
public:
    static WorkerPool& getInstance(size_t max_threads = std::thread::hardware_concurrency() * 2) {
        static WorkerPool instance(max_threads);
        return instance;
    }

    ~WorkerPool() { shutdown(); }

    std::shared_ptr<_TaskInterface> getTaskBlocking() { // Renamed for clarity
        std::unique_lock<std::mutex> lock(lock_);
        cv_.wait(lock, [this] { return !task_queue_.empty() || shutting_down_; });
        if (shutting_down_ && task_queue_.empty()) { return nullptr; }
        if (task_queue_.empty()) { return nullptr; } // Can happen if woken by shutdown
        auto task = task_queue_.top();
        task_queue_.pop();
        return task;
    }
    
    std::shared_ptr<_TaskInterface> tryGetTask() {
        std::unique_lock<std::mutex> lock(lock_);
        if (task_queue_.empty() || shutting_down_) {
            return nullptr;
        }
        auto task = task_queue_.top();
        task_queue_.pop();
        return task;
    }

    std::shared_ptr<_TaskInterface> getTask() { // Called by worker threads
        std::unique_lock<std::mutex> lock(lock_);
        if (task_queue_.empty() || shutting_down_) { return nullptr; }
        auto task = task_queue_.top();
        task_queue_.pop();
        return task;
    }

    void shutdown() {
        if (shutting_down_.exchange(true)) { return; }
        cv_.notify_all(); // Wake up all workers
        workers_.clear(); // Clears the vector, destroying unique_ptrs, which destroys _Workers
    }

    void submitTask(std::shared_ptr<_TaskInterface> task) {
        std::unique_lock<std::mutex> lock(lock_);
        task_queue_.push(task);
        cv_.notify_one();
    }

    bool is_shutting_down() const { return shutting_down_; }

private: 
    std::mutex lock_;
    std::vector<_Worker> workers_;
    std::atomic<bool> shutting_down_ = false;
    std::condition_variable cv_;
    size_t running_thread_max_ = std::thread::hardware_concurrency() * 2;
    std::priority_queue<std::shared_ptr<_TaskInterface>, std::vector<std::shared_ptr<_TaskInterface>>, _TaskPtrComparator> task_queue_;
    WorkerPool(size_t max_threads)  // Private constructor for singleton
        : running_thread_max_(max_threads) {
        workers_.reserve(running_thread_max_);
        for (size_t i = 0; i < running_thread_max_; ++i) { workers_.emplace_back(this); }
    }
};

class _TaskInterface : public std::enable_shared_from_this<_TaskInterface> {
public: 
    enum class State { PENDING, RUNNING, DONE, FAILED };

    static thread_local _TaskInterface* current_task_tls_; // Definition in .cpp or inline static
    std::weak_ptr<_TaskInterface> parent_weak_ptr_;
    
    bool is_registered_as_child_;
    std::vector<std::shared_ptr<_TaskInterface>> children_;
    std::mutex children_mutex_;

    bool operator<  (const _TaskInterface& other) const {
        return this->priority < other.priority ||
              (this->priority == other.priority && this->id_.length() < other.id_.length()) ||
              (this->priority == other.priority && this->id_.length() == other.id_.length() && this->birth_time_ < other.birth_time_);
    }
    bool operator>  (const _TaskInterface& other) const {
        return this->priority > other.priority ||
              (this->priority == other.priority && this->id_.length() > other.id_.length()) ||
              (this->priority == other.priority && this->id_.length() == other.id_.length() && this->birth_time_ > other.birth_time_);
    }
    bool operator== (const _TaskInterface& other) const { return this->id_ == other.id_; }
    
    void addChild(std::shared_ptr<_TaskInterface> child_task) {
        std::lock_guard<std::mutex> lock(children_mutex_);
        children_.push_back(child_task);
    }

    virtual void execute() = 0;  // Worker thread calls this
    virtual void join()    = 0;  // Wait for task completion

protected: 
    WorkerPool& worker_pool_ = WorkerPool::getInstance();
    int64_t priority = 0;
    std::string id_;
    std::string name_;
    State      state_ = State::PENDING;  // Default state
    std::mutex task_mutex_;              // To protect task-specific state if needed
    std::condition_variable task_cv_;     // Condition variable for task completion
    
    std::chrono::time_point<std::chrono::system_clock> birth_time_;
    std::chrono::time_point<std::chrono::system_clock> start_time_;
    std::chrono::time_point<std::chrono::system_clock> end_time_;

    _TaskInterface() : is_registered_as_child_(false) 
    {
        if (_TaskInterface::current_task_tls_) {
            parent_weak_ptr_ = _TaskInterface::current_task_tls_->shared_from_this();
        }
        birth_time_ = std::chrono::system_clock::now();
        if (auto parent = parent_weak_ptr_.lock()) { // Use the member name
            id_ = parent->generateNewChildID();
        } else { id_ = TaskUtils::generateUUID(); }
    }

    void waitForChildren() {
        std::vector<std::shared_ptr<_TaskInterface>> current_children;
        {
            std::lock_guard<std::mutex> lock(children_mutex_);
            if (children_.empty()) return;
            current_children = children_; // Copy to release lock while waiting
        }
        for (const auto& child : current_children) { child->join(); }
    }

    _TaskInterface  (const _TaskInterface&) = delete;
    _TaskInterface  (_TaskInterface&&)      = delete;
    _TaskInterface& operator                = (const _TaskInterface&) = delete;
    
    std::string getID() const { return id_; }
    std::string getName() const { return name_; }
    virtual std::string generateNewChildID() { std::string new_id = id_ + TaskUtils::generateRandomHex(4); return new_id; }

    void setStartTime(std::chrono::time_point<std::chrono::system_clock> start_time) { start_time_ = start_time; }
    void setEndTime(std::chrono::time_point<std::chrono::system_clock> end_time) { end_time_ = end_time; }

    std::chrono::time_point<std::chrono::system_clock> getBirthTime() const { return birth_time_; }
    std::chrono::time_point<std::chrono::system_clock> getStartTime() const { return start_time_; }
    std::chrono::time_point<std::chrono::system_clock> getEndTime() const { return end_time_; }
    std::chrono::duration<double> getElapsedTime() const { return std::chrono::system_clock::now() - birth_time_; }
    std::chrono::duration<double> getExecutionTime() const { return end_time_ - start_time_; }
};

class ScopedCurrentTaskSetter {
    _TaskInterface* previous_task_in_tls_;
public:
    ScopedCurrentTaskSetter(_TaskInterface* task_to_set_as_current)
        : previous_task_in_tls_(_TaskInterface::current_task_tls_) {
        _TaskInterface::current_task_tls_ = task_to_set_as_current;
    }
    ~ScopedCurrentTaskSetter() { _TaskInterface::current_task_tls_ = previous_task_in_tls_; }
    ScopedCurrentTaskSetter(const ScopedCurrentTaskSetter&) = delete;
    ScopedCurrentTaskSetter& operator=(const ScopedCurrentTaskSetter&) = delete;
};

template<typename _T>
class _TaskBase : public _TaskInterface {
protected: 
    virtual void execute_impl() = 0;

    public: 
    void execute() override final {
        ScopedCurrentTaskSetter scoped_setter(this);
        this->setStartTime(std::chrono::system_clock::now());
        
        std::unique_lock<std::mutex> lock(this->task_mutex_);
        this->state_ = State::RUNNING;
        this->task_cv_.notify_all();  // Notify any waiting threads
        lock.unlock();

        try {
            static_cast<_T*>(this)->execute_impl();
            std::unique_lock<std::mutex> lock(this->task_mutex_);
            state_ = State::DONE;
        } catch (...) {
            std::unique_lock<std::mutex> lock(this->task_mutex_);
            state_ = State::FAILED;
        }
        this->waitForChildren();  // Wait for child tasks to finish


        this->setEndTime(std::chrono::system_clock::now());
        {
            std::unique_lock<std::mutex> lock(this->task_mutex_); // Not strictly needed if state is final
            this->task_cv_.notify_all(); // Notify any joining threads
        }
    }
};

template<typename Result, typename... Args>
class Task : public _TaskBase<Task<Result, Args...>> {
private:
    std::function<Result    (Args...)> task_function_;                 // Function to execute
    std::tuple<Args...>     args_;                                     // Arguments to pass to the function
    std::condition_variable task_cv_;                                  // Condition variable for task completion
    std::promise<Result>    result_promise_;                           // Promise to set the result
    std::future<Result>     result_future_;                            // Future to retrieve the result
    
public: 
    Task(std::function<Result(Args...)>&& func, const std::tuple<Args...>& args, const std::string& task_name = "", int64_t task_priority = 0)
    : task_function_(std::move(func)), args_(args), result_future_(result_promise_.get_future()) {
        this->name_    = task_name;
        this->priority = task_priority;
    }

    // CRTP methods called by _TaskBase::execute()
    void execute_impl() override {
        try {
            result_promise_.set_value(std::apply(task_function_, args_));
        } catch (...) {
            result_promise_.set_exception(captured_exception_);
        }
    }

    void start() {
        // Register with parent if applicable
        if (auto parent_sp = this->parent_weak_ptr_.lock()) { // Lock the weak_ptr (assuming you rename parent_weak_ptr_)
                if (!this->is_registered_as_child_) {
                    parent_sp->addChild(this->shared_from_this());
                    this->is_registered_as_child_ = true;
                }
        }
        // Add to global task map
        {
            std::lock_guard<std::mutex> lock(all_tasks_mutex_);
            all_tasks_[this->getID()] = this->shared_from_this();
        }
        worker_pool_.submitTask(this->shared_from_this());
    }

    void join() override {
        if (current_worker_) {
            while (result_future_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                if(!current_worker_->try_poll_once()) {std::this_thread::yield();}
        } } else { result_future_.wait(); }
        if (captured_exception_) { std::rethrow_exception(captured_exception_); }
        {
            std::lock_guard<std::mutex> lock(all_tasks_mutex_);
            all_tasks_.erase(this->getID());
        }
    }

    Result getResult() {
        if (result_future_.valid()) {
            return result_future_.get();
        } else { throw TaskException("Task result is not valid"); }
    }
};  // class Task


template<typename = void, typename... Args>
class Task : public _TaskBase<Task<void, Args...>> {
private: 
    std::function<void     (Args...)> task_function_;  // Function to execute
    std::tuple<Args...>     args_;                     // Arguments to pass to the function
    std::condition_variable task_cv_;                  // Condition variable for task completion
    std::promise<void>      result_promise_;           // Promise to set the result
    std::future<void>       result_future_;            // Future to retrieve the result
public: 
    Task(std::function<void(Args...)>&& func, const std::tuple<Args...>& args, const std::string& task_name = "", int64_t task_priority = 0)
    : task_function_(std::move(func)), args_(args), result_future_(result_promise_.get_future()) {
        this->name_    = task_name;
        this->priority = task_priority;
    }

    // CRTP methods called by _TaskBase::execute()
    void execute_impl() override {
        try {
            std::apply(task_function_, args_);
            result_promise_.set_value();
        } catch (...) {
            result_promise_.set_exception(std::current_exception());
        }
    }

    void start() {
        // Register with parent if applicable
        if (auto parent_sp = this->parent_weak_ptr_.lock()) { // Lock the weak_ptr (assuming you rename parent_weak_ptr_)
            if (!this->is_registered_as_child_) {
                    parent_sp->addChild(this->shared_from_this());
                    this->is_registered_as_child_ = true;
                }
        }
        // Add to global task map
        {
            std::lock_guard<std::mutex> lock(all_tasks_mutex_);
            all_tasks_[this->getID()] = this->shared_from_this();
        }
        worker_pool_.submitTask(this->shared_from_this());
    }

    void join() override {
        if (current_worker_) {
            while (result_future_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                if(!current_worker_->try_poll_once()) {std::this_thread::yield();}
        } else { result_future_.wait(); }
        if (captured_exception_) { std::rethrow_exception(captured_exception_); }
        {
            std::lock_guard<std::mutex> lock(all_tasks_mutex_);
            all_tasks_.erase(this->getID());
        }
        }
    }
    
    void getResult() {
        if (result_future_.valid()) { result_future_.get();
        } else { throw TaskException("Task result is not valid"); }
    }
};  // class Task

}; // namespace thread
#endif // __TASK_H__