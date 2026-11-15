#ifndef __PATH_TREE_H__
#define __PATH_TREE_H__

#include "utils/utils.h"
#include "utils/task.h"
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream> // For error logging
#include <stdexcept> // For exceptions

namespace utils::system {

namespace fs = std::filesystem;
class PathNode;
class PathTree;

class PathNode : public std::enable_shared_from_this<PathNode> {
public:
    enum class NodeType { // IMPROVEMENT: Renamed Type to NodeType for clarity
        FILE,
        LINK,
        DIRECTORY,
        OTHER,
        UNINITIALIZED // IMPROVEMENT: Added for nodes not yet parsed
    };

protected:
    std::string name_; // IMPROVEMENT: Suffix underscore for private members
    std::weak_ptr<PathNode> parent_;
    std::weak_ptr<PathTree> tree_head_; // IMPROVEMENT: Renamed for clarity
    int depth_;
    int depth_limit_;
    std::uintmax_t size_bytes_; // IMPROVEMENT: Renamed for clarity
    NodeType node_type_; // IMPROVEMENT: Cache node type

    std::vector<std::shared_ptr<PathNode>> children_;
    mutable std::mutex node_mutex_; // Mutex for children_ and other mutable shared state

public:
    PathNode(const std::string& node_name,
             std::shared_ptr<PathNode> parent_node = nullptr,
             std::shared_ptr<PathTree> tree_head = nullptr, // IMPROVEMENT: Pass tree_head explicitly
             int current_depth = 0, // IMPROVEMENT: Pass depth explicitly
             int limit = 10);

    virtual ~PathNode() = default;

    // --- Informational ---
    virtual std::string getFullPath() const; // IMPROVEMENT: Consistently use getFullPath
    std::string getName() const { return name_; }
    std::uintmax_t getSizeBytes() const; // IMPROVEMENT: Made const, potentially reads from cache
    int getDepth() const { return depth_; }
    NodeType getNodeType(); // IMPROVEMENT: Lazily determines type and caches it
    const std::vector<std::shared_ptr<PathNode>>& getChildren() const; // IMPROVEMENT: Made const, returns const ref

    bool isFile() { return getNodeType() == NodeType::FILE; }
    bool isDirectory() { return getNodeType() == NodeType::DIRECTORY; }
    bool isSymbolicLink() { return getNodeType() == NodeType::LINK; } // IMPROVEMENT: Renamed
    bool childrenLoaded() const; // IMPROVEMENT: Check if children have been populated by parse

    // --- Comparison & String ---
    virtual std::string toString(int max_children_to_display = 4) const;
    bool operator==(const std::string& other_name) const { return name_ == other_name; }
    bool operator==(const PathNode& other) const { return name_ == other.name_ && getFullPath() == other.getFullPath(); } // IMPROVEMENT: Compare full path too for more robust equality
    bool operator>(const PathNode& other) const { return name_ > other.name_; }
    bool operator<(const PathNode& other) const { return name_ < other.name_; }

    // --- Child Management (In-memory representation) ---
    std::shared_ptr<PathNode> getChild(const std::string& child_name) const;
    bool removeChildFromMemory(const std::shared_ptr<PathNode>& child_node); // IMPROVEMENT: Clearer name

    // --- Filesystem Operations ---
    std::shared_ptr<PathNode> createDirectory(const std::string& dir_name); // IMPROVEMENT: Renamed mkdir
    std::shared_ptr<PathNode> createFile(const std::string& file_name);    // IMPROVEMENT: Renamed touch
    void moveTo(std::shared_ptr<PathNode> destination_directory); // IMPROVEMENT: Clearer name
    void deleteFromFilesystem(bool recursive = true); // IMPROVEMENT: Clearer name, explicit recursion for directories

    // --- Traversal & Searching ---
    void populateChildren(bool recursive = false); // IMPROVEMENT: Explicitly populates children (was parse)
    std::vector<std::shared_ptr<PathNode>> findDescendants(const std::string& pattern, bool exact_match = false); // IMPROVEMENT: Clearer find

    // --- Iterator ---
    class DescendantFileIterator { // IMPROVEMENT: More descriptive name
    private:
        std::vector<std::shared_ptr<PathNode>> collected_nodes_;
        size_t current_index_;
        void collectNodesRecursively(const std::shared_ptr<PathNode>& node, bool files_only);
    public:
        // IMPROVEMENT: Constructor now takes a flag for files_only or all nodes
        explicit DescendantFileIterator(const std::shared_ptr<PathNode>& root_node, bool files_only = true);
        DescendantFileIterator(); // End iterator

        DescendantFileIterator& operator++();
        std::shared_ptr<PathNode> operator*() const;
        bool operator==(const DescendantFileIterator& other) const;
        bool operator!=(const DescendantFileIterator& other) const;

        // C++17 iterator traits (optional but good practice)
        using iterator_category = std::input_iterator_tag;
        using value_type = std::shared_ptr<PathNode>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
    };
    DescendantFileIterator begin_files(); // Iterates only files
    DescendantFileIterator end_files();
    // Could add begin_all_nodes() / end_all_nodes() if needed

protected:
    // --- Implementation Details ---
    void updateCachedInfoFromFilesystem(); // IMPROVEMENT: Central place to get type/size
    void populateDirectoryChildrenImpl();
    void moveToImpl(std::shared_ptr<PathNode> destination_directory);
    void deleteFromFilesystemImpl(bool recursive);
    std::vector<std::shared_ptr<PathNode>> findDescendantsImpl(const std::string& pattern, bool exact_match);

    // Helper to add child and set its parent/head correctly
    void addChildToMemory(std::shared_ptr<PathNode> child);
};

class PathTree : public PathNode {
public:
    explicit PathTree(const std::string& root_path_str, int parse_depth_limit = 5);
    std::string getPath() const { return getFullPath(); } // IMPROVEMENT: Consistent naming
    std::string getFullPath() const override; // PathTree's name *is* the full path
};


// --- PathNode Implementation ---
PathNode::PathNode(const std::string& node_name,
                   std::shared_ptr<PathNode> parent_node,
                   std::shared_ptr<PathTree> tree_head_ptr,
                   int current_depth,
                   int limit)
    : name_(node_name),
      depth_(current_depth),
      depth_limit_(limit),
      size_bytes_(0), // Will be populated by populateChildren or updateCachedInfo
      node_type_(NodeType::UNINITIALIZED) {
    if (parent_node) {
        parent_ = parent_node;
    }
    if (tree_head_ptr) {
        tree_head_ = tree_head_ptr;
    }
    // IMPROVEMENT: If parent_node is null, this node might be the tree_head itself.
    // This is handled in PathTree constructor.
}

std::string PathNode::getFullPath() const {
    // IMPROVEMENT: More robust path construction
    if (auto p_ptr = parent_.lock()) {
        fs::path parent_path = p_ptr->getFullPath();
        return (parent_path / name_).lexically_normal().string();
    }
    // If no parent, this node is likely the root (PathTree), or an orphaned node.
    // PathTree overrides this to just return its name (which is an absolute path).
    // For a generic PathNode without a parent, its name is its "path".
    return name_;
}

PathNode::NodeType PathNode::getNodeType() {
    // IMPROVEMENT: Lazy loading and caching of node type
    std::lock_guard<std::mutex> lock(node_mutex_); // Protects node_type_
    if (node_type_ == NodeType::UNINITIALIZED) {
        updateCachedInfoFromFilesystem();
    }
    return node_type_;
}

std::uintmax_t PathNode::getSizeBytes() const {
    // IMPROVEMENT: If size is calculated during populate, it should be accurate.
    // This getter just returns the cached value.
    // Consider if it should trigger a filesystem read if not populated.
    std::lock_guard<std::mutex> lock(node_mutex_);
    return size_bytes_;
}


void PathNode::updateCachedInfoFromFilesystem() {
    // Called under node_mutex_ lock
    fs::path path = getFullPath();
    std::error_code ec;

    if (!fs::exists(path, ec) || ec) {
        node_type_ = NodeType::OTHER; // Or some "NotFound" state
        size_bytes_ = 0;
        // std::cerr << "PathNode Error: Path does not exist or error checking existence: " << path.string() << " - " << ec.message() << std::endl;
        return;
    }

    if (fs::is_directory(path, ec)) {
        node_type_ = NodeType::DIRECTORY;
        // Size of directory is sum of children, calculated during populateChildren
    } else if (fs::is_regular_file(path, ec)) {
        node_type_ = NodeType::FILE;
        size_bytes_ = fs::file_size(path, ec);
        if (ec) {
            // std::cerr << "PathNode Error: Could not get file size for " << path.string() << " - " << ec.message() << std::endl;
            size_bytes_ = 0;
        }
    } else if (fs::is_symlink(path, ec)) {
        node_type_ = NodeType::LINK;
        // Size of symlink itself is usually small, or could be target's size (more complex)
        // For now, let's get symlink's own size if possible, or 0
        // fs::file_size on a symlink might give target size or error depending on OS.
        // To be safe, let's try, but default to 0 on error.
        std::uintmax_t link_size = fs::file_size(path, ec);
        if (!ec) {
            size_bytes_ = link_size;
        } else {
            size_bytes_ = 0; // Keep it simple for now
        }

    } else {
        node_type_ = NodeType::OTHER;
        size_bytes_ = 0;
    }
    if (ec) { // Catch-all for errors from fs::is_... calls
        // std::cerr << "PathNode Error: Filesystem check error for " << path.string() << " - " << ec.message() << std::endl;
        node_type_ = NodeType::OTHER; // Revert to OTHER on error
        size_bytes_ = 0;
    }
}

std::string PathNode::toString(int max_children_to_display) const {
    std::lock_guard<std::mutex> lock(node_mutex_); // Protect children_ access
    std::stringstream ss;
    ss << std::string(depth_ * 2, ' ') << "|-- " << name_;

    // Access type and size through getters to ensure they are initialized if needed
    // (though toString is const, getNodeType might initialize if not already)
    NodeType current_type = const_cast<PathNode*>(this)->getNodeType(); // Hack to call non-const getNodeType
    std::uintmax_t current_size = size_bytes_; // Use cached size

    std::string type_str;
    switch (current_type) {
        case NodeType::DIRECTORY: type_str = " (DIR)"; break;
        case NodeType::FILE: type_str = " (FILE)"; break;
        case NodeType::LINK: type_str = " (LINK)"; break;
        default: type_str = " (OTHER)"; break;
    }
    ss << type_str;

    int len_interval = 60 - (depth_ * 2 + 4 + name_.length() + type_str.length());
    ss << std::string(std::max(0, len_interval), ' ') << bytesToHumanReadable(current_size) << "\n";

    if (current_type == NodeType::DIRECTORY && !children_.empty()) {
        int count = 0;
        for (const auto& child : children_) {
            if (count >= max_children_to_display) {
                ss << std::string((depth_ + 1) * 2, ' ') << "|-- ... (" << children_.size() - count << " more)\n";
                break;
            }
            ss << child->toString(max_children_to_display); // Recursive call
            count++;
        }
    }
    return ss.str();
}


bool PathNode::childrenLoaded() const {
    std::lock_guard<std::mutex> lock(node_mutex_);
    // A simple heuristic: if it's a directory and children_ is empty,
    // but it hasn't been explicitly marked as NodeType::DIRECTORY yet,
    // it's likely not loaded. A more robust way would be a separate flag.
    return node_type_ == NodeType::DIRECTORY && !children_.empty(); // Or node_type_ != UNINITIALIZED for files/links
}

const std::vector<std::shared_ptr<PathNode>>& PathNode::getChildren() const {
    // IMPROVEMENT: No need to call populate here. Let user call populateChildren explicitly.
    // If a const method is needed that ensures population, it's tricky.
    std::lock_guard<std::mutex> lock(node_mutex_);
    return children_;
}

std::shared_ptr<PathNode> PathNode::getChild(const std::string& child_name) const {
    std::lock_guard<std::mutex> lock(node_mutex_);
    for (const auto& child : children_) {
        if (child->name_ == child_name) {
            return child;
        }
    }
    return nullptr;
}

bool PathNode::removeChildFromMemory(const std::shared_ptr<PathNode>& child_node) {
    std::lock_guard<std::mutex> lock(node_mutex_);
    auto it = std::find(children_.begin(), children_.end(), child_node);
    if (it != children_.end()) {
        children_.erase(it);
        return true;
    }
    return false;
}

void PathNode::addChildToMemory(std::shared_ptr<PathNode> child) {
    std::lock_guard<std::mutex> lock(node_mutex_);
    children_.push_back(child);
    // Parent and head should have been set in child's constructor
}


std::shared_ptr<PathNode> PathNode::createDirectory(const std::string& dir_name) {
    if (getNodeType() != NodeType::DIRECTORY && node_type_ != NodeType::UNINITIALIZED) {
         // std::cerr << "PathNode Error: Cannot create directory in a non-directory node: " << getFullPath() << std::endl;
        return nullptr; // Or throw
    }
    if (depth_ + 1 > depth_limit_){
        // std::cerr << "PathNode Error: Depth limit reached. Cannot create directory: " << dir_name << std::endl;
        return nullptr;
    }

    fs::path new_dir_fs_path = fs::path(getFullPath()) / dir_name;
    std::error_code ec;
    if (!fs::create_directory(new_dir_fs_path, ec)) {
        // std::cerr << "PathNode Error: Filesystem failed to create directory " << new_dir_fs_path.string() << " - " << ec.message() << std::endl;
        return nullptr; // Or throw
    }

    auto new_node = std::make_shared<PathNode>(dir_name, shared_from_this(), tree_head_.lock(), depth_ + 1, depth_limit_);
    new_node->node_type_ = NodeType::DIRECTORY; // We just created it
    addChildToMemory(new_node);
    return new_node;
}

std::shared_ptr<PathNode> PathNode::createFile(const std::string& file_name) {
    if (getNodeType() != NodeType::DIRECTORY && node_type_ != NodeType::UNINITIALIZED) {
        // std::cerr << "PathNode Error: Cannot create file in a non-directory node: " << getFullPath() << std::endl;
        return nullptr;
    }
     if (depth_ + 1 > depth_limit_){
        // std::cerr << "PathNode Error: Depth limit reached. Cannot create file: " << file_name << std::endl;
        return nullptr;
    }

    fs::path new_file_fs_path = fs::path(getFullPath()) / file_name;
    std::ofstream file_stream(new_file_fs_path.string());
    if (!file_stream) {
        // std::cerr << "PathNode Error: Filesystem failed to create file " << new_file_fs_path.string() << std::endl;
        return nullptr;
    }
    file_stream.close();

    auto new_node = std::make_shared<PathNode>(file_name, shared_from_this(), tree_head_.lock(), depth_ + 1, depth_limit_);
    new_node->node_type_ = NodeType::FILE; // We just created it
    new_node->size_bytes_ = 0;
    addChildToMemory(new_node);
    return new_node;
}

void PathNode::populateChildren(bool recursive) {
    // Ensure this node's type is known and it's a directory
    if (getNodeType() != NodeType::DIRECTORY) {
        return; // Only directories can have children populated
    }
    if (depth_ >= depth_limit_ && recursive) { // Check depth limit for recursion
        return;
    }

    // Use a task to run the implementation
    // For simplicity here, making it synchronous. An async version would return the task.
    utils::task::Task<void> task([this, recursive]() {
        this->populateDirectoryChildrenImpl();
        if (recursive) {
            std::vector<std::shared_ptr<PathNode>> current_children_copy;
            {
                std::lock_guard<std::mutex> lock(node_mutex_);
                current_children_copy = children_; // Copy to iterate without holding lock
            }
            std::vector<utils::task::Task<void>> child_tasks;
            for (const auto& child : current_children_copy) {
                // Check depth limit before recursing for each child
                if (child->depth_ < child->depth_limit_) {
                     child_tasks.emplace_back([child](){ child->populateChildren(true); }, std::make_tuple());
                     child_tasks.back().start();
                } else if (child->isDirectory()){ // If it's a directory at depth limit, at least get its basic info
                    child_tasks.emplace_back([child](){ child->getNodeType(); child->getSizeBytes(); }, std::make_tuple());
                    child_tasks.back().start();
                }
            }
            for(auto& ct : child_tasks) ct.join();
        }
        // After all children (and their sub-children if recursive) are populated, update this directory's size
        std::uintmax_t total_size = 0;
        std::lock_guard<std::mutex> lock(node_mutex_);
        for(const auto& child : children_){
            total_size += child->size_bytes_; // Assumes child sizes are now correct
        }
        size_bytes_ = total_size;

    }, std::make_tuple());
    task.start();
    task.join();
}

void PathNode::populateDirectoryChildrenImpl() {
    // This is the core logic executed by the task in populateChildren
    std::lock_guard<std::mutex> lock(node_mutex_); // Protects children_ and node_type_
    if (node_type_ != NodeType::DIRECTORY) return; // Should have been checked before

    children_.clear(); // Clear existing children before re-populating
    size_bytes_ = 0;   // Reset size, will be recalculated

    fs::path dir_fs_path = getFullPath();
    std::error_code ec;

    if (!fs::exists(dir_fs_path, ec) || !fs::is_directory(dir_fs_path, ec) || ec) {
        // std::cerr << "PathNode Error: Cannot populate children, path is not a valid directory: " << dir_fs_path.string() << " - " << ec.message() << std::endl;
        node_type_ = NodeType::OTHER; // Mark as other if it's no longer a directory
        return;
    }

    for (const auto& entry : fs::directory_iterator(dir_fs_path, fs::directory_options::skip_permission_denied, ec)) {
        if (ec) {
            // std::cerr << "PathNode Warning: Error iterating directory " << dir_fs_path.string() << " - " << ec.message() << std::endl;
            continue;
        }
        // Create child node without immediate deep parsing.
        // Its type and size will be fetched when getNodeType/getSizeBytes is called, or if populateChildren(true) is called on it.
        auto child_node = std::make_shared<PathNode>(
            entry.path().filename().string(),
            shared_from_this(),
            tree_head_.lock(),
            depth_ + 1,
            depth_limit_
        );
        // No need to call child_node->updateCachedInfoFromFilesystem() here, it will be done lazily or by recursive populate.
        children_.push_back(child_node);
    }
     // Sort children by name for consistent order
    std::sort(children_.begin(), children_.end(),
              [](const std::shared_ptr<PathNode>& a, const std::shared_ptr<PathNode>& b) {
                  return a->name_ < b->name_;
              });
}

void PathNode::moveTo(std::shared_ptr<PathNode> destination_directory) {
    if (!destination_directory || (destination_directory->getNodeType() != NodeType::DIRECTORY && destination_directory->node_type_ != NodeType::UNINITIALIZED)) {
        throw utils::task::TaskException("Destination for move must be a valid directory.");
    }
    // Prevent moving a directory into itself or its own children
    auto current = destination_directory;
    while(auto p = current->parent_.lock()){
        if(p.get() == this) throw utils::task::TaskException("Cannot move a directory into itself or one of its subdirectories.");
        current = p;
    }
    if(destination_directory.get() == this) throw utils::task::TaskException("Cannot move a directory into itself.");

    utils::task::Task<void> task([this, destination_directory]() {
        this->moveToImpl(destination_directory);
    }, std::make_tuple());
    task.start();
    task.join(); // Make public API synchronous for now
}

void PathNode::moveToImpl(std::shared_ptr<PathNode> destination_directory) {
    fs::path src_fs_path = getFullPath();
    fs::path dst_parent_fs_path = destination_directory->getFullPath();
    fs::path dst_fs_path = dst_parent_fs_path / name_;
    std::error_code ec;

    // Handle existing item at destination
    if (fs::exists(dst_fs_path, ec)) {
        // Simple strategy: remove existing before moving. Could be configurable.
        // More complex: merge if both are directories (as you had before).
        // For now, let's keep it simpler: fail if destination exists and is different type, or remove if same.
        PathNode temp_dest_node(name_, destination_directory, tree_head_.lock(), destination_directory->depth_ + 1, depth_limit_);
        temp_dest_node.updateCachedInfoFromFilesystem(); // Check what's actually there

        if (this->getNodeType() == NodeType::DIRECTORY && temp_dest_node.getNodeType() == NodeType::DIRECTORY) {
            // MERGE: This is complex. For simplicity, let's disallow direct overwrite merge for now
            // or require the destination directory to be empty.
            // A true merge would involve moving children one by one.
            //  std::cerr << "PathNode Warning: Destination directory " << dst_fs_path.string() << " already exists. Merge not implemented, aborting move." << std::endl;
             // Alternatively, could implement a recursive move of children here.
             // For now, let's assume it's an error to move a dir on top of another non-empty dir
             // or implement a clear then move.
             // To implement merge:
            for (auto& child_to_move : this->children_) {
            child_to_move->moveToImpl(destination_directory->getChild(this->name_)); // Assuming dest dir node is created
            }
            fs::remove(src_fs_path, ec); // remove original empty shell
        }
        // If not dir-to-dir merge, remove destination first
        fs::remove_all(dst_fs_path, ec); // remove_all for directories
        if (ec) {
            std::cerr << "PathNode Error: Failed to remove existing item at destination " << dst_fs_path.string() << " - " << ec.message() << std::endl;
            return;
        }
    }

    fs::rename(src_fs_path, dst_fs_path, ec);
    if (ec) {
        std::cerr << "PathNode Error: Filesystem rename failed from " << src_fs_path.string() << " to " << dst_fs_path.string() << " - " << ec.message() << std::endl;
        return;
    }

    // Update in-memory tree structure
    auto old_parent_ptr = parent_.lock();
    if (old_parent_ptr) {
        old_parent_ptr->removeChildFromMemory(shared_from_this());
    }

    parent_ = destination_directory;
    depth_ = destination_directory->depth_ + 1;
    // Tree head should remain the same unless moving between different conceptual trees (not supported here)
    destination_directory->addChildToMemory(shared_from_this());

    // Recursively update paths and depths of children if any (though children vector is not changed by rename)
    // This is important if getFullPath() relies on cached full paths rather than dynamic construction.
    // Since our getFullPath is dynamic, only depth needs propagation if children are already parsed.
    std::vector<std::shared_ptr<PathNode>> children_copy;
    {
        std::lock_guard<std::mutex> lock(node_mutex_);
        children_copy = children_;
    }
    for(const auto& child : children_copy) {
        // Simple depth update. Full path will be reconstructed correctly.
        std::function<void(std::shared_ptr<PathNode>, int)> update_depths_recursive =
            [&](std::shared_ptr<PathNode> node, int new_parent_depth) {
            node->depth_ = new_parent_depth + 1;
            std::vector<std::shared_ptr<PathNode>> node_children_copy;
            {
                std::lock_guard<std::mutex> child_lock(node->node_mutex_);
                node_children_copy = node->children_;
            }
            for(const auto& c : node_children_copy){
                update_depths_recursive(c, node->depth_);
            }
        };
        update_depths_recursive(child, this->depth_);
    }
}

void PathNode::deleteFromFilesystem(bool recursive) {
    utils::task::Task<void> task([this, recursive]() {
        this->deleteFromFilesystemImpl(recursive);
    }, std::make_tuple());
    task.start();
    task.join();
}

void PathNode::deleteFromFilesystemImpl(bool recursive) {
    fs::path path_to_remove = getFullPath();
    std::error_code ec;

    if (getNodeType() == NodeType::DIRECTORY) {
        if (!recursive) {
             // Check if directory is empty using filesystem, not in-memory children_
            bool is_empty = true;
            std::error_code iter_ec;
            for (const auto& entry : fs::directory_iterator(path_to_remove, fs::directory_options::skip_permission_denied, iter_ec)) {
                if (iter_ec) { /* handle error, maybe throw or log */ break; }
                is_empty = false;
                break;
            }
            if (iter_ec) { /* handle error from directory_iterator constructor */ }

            if (!is_empty) {
                throw utils::task::TaskException("Cannot delete non-empty directory without recursive flag: " + path_to_remove.string());
            }
        }
        // If recursive, or if empty (and non-recursive), fs::remove_all is safe.
        fs::remove_all(path_to_remove, ec);
    } else if (node_type_ == NodeType::FILE || node_type_ == NodeType::LINK) {
        fs::remove(path_to_remove, ec);
    } else if (node_type_ == NodeType::UNINITIALIZED || node_type_ == NodeType::OTHER) {
        // Attempt to remove, might be a broken symlink or something OS can remove
        if(fs::exists(path_to_remove, ec) || fs::is_symlink(fs::symlink_status(path_to_remove))) { // Check existence or if it's a symlink (even broken)
            fs::remove(path_to_remove, ec);
        } else {
            // std::cout << "PathNode Info: Item to delete does not exist or is of unknown type: " << path_to_remove.string() << std::endl;
        }
    }

    if (ec) {
        throw utils::task::TaskException("Filesystem error deleting " + path_to_remove.string() + ": " + ec.message());
    }

    // Update in-memory tree
    auto parent_ptr = parent_.lock();
    if (parent_ptr) {
        parent_ptr->removeChildFromMemory(shared_from_this());
    }
    // Node is now "dead"
    node_type_ = NodeType::OTHER; // Or a new "DELETED" state
    children_.clear(); // If it was a directory
    size_bytes_ = 0;
}


std::vector<std::shared_ptr<PathNode>> PathNode::findDescendants(const std::string& pattern, bool exact_match) {
    utils::task::Task<std::vector<std::shared_ptr<PathNode>>> task([this, pattern, exact_match]() {
        return this->findDescendantsImpl(pattern, exact_match);
    }, std::make_tuple());
    task.start();
    return task.getResult(); // Join happens in getResult if your Task works that way
}

std::vector<std::shared_ptr<PathNode>> PathNode::findDescendantsImpl(const std::string& pattern, bool exact_match) {
    std::vector<std::shared_ptr<PathNode>> results;
    std::vector<utils::task::Task<std::vector<std::shared_ptr<PathNode>>>> child_search_tasks;

    // Ensure children are loaded for the current node if it's a directory
    if (this->getNodeType() == NodeType::DIRECTORY && !this->childrenLoaded()) {
        // Non-recursive populate for current level only for find.
        // Or, the user should call populateChildren first for deep searches.
        // For find, it's better to populate as we go if not already done.
        this->populateDirectoryChildrenImpl(); // Just this level
    }

    std::vector<std::shared_ptr<PathNode>> current_children_copy;
    {
        std::lock_guard<std::mutex> lock(node_mutex_);
        current_children_copy = children_;
    }

    for (const auto& child : current_children_copy) {
        bool matches = false;
        if (exact_match) {
            matches = (child->name_ == pattern);
        } else {
            matches = (child->name_.find(pattern) != std::string::npos);
        }

        if (matches) {
            results.push_back(child);
        }

        if (child->isDirectory() && child->depth_ < child->depth_limit_) { // Only recurse into directories within depth limit
             child_search_tasks.emplace_back(
                [child, pattern, exact_match]() { return child->findDescendantsImpl(pattern, exact_match); },
                std::make_tuple());
            child_search_tasks.back().start();
        }
    }

    for (auto& child_task : child_search_tasks) {
        auto task_results = child_task.getResult(); // Assumes getResult joins
        results.insert(results.end(),
                       std::make_move_iterator(task_results.begin()),
                       std::make_move_iterator(task_results.end()));
    }
    return results;
}


// --- Iterator Implementation ---
PathNode::DescendantFileIterator::DescendantFileIterator(const std::shared_ptr<PathNode>& root_node, bool files_only)
    : current_index_(0) {
    if (root_node) {
        collectNodesRecursively(root_node, files_only);
    }
}

PathNode::DescendantFileIterator::DescendantFileIterator() : current_index_(0) {} // End iterator

void PathNode::DescendantFileIterator::collectNodesRecursively(const std::shared_ptr<PathNode>& node, bool files_only) {
    if (!node) return;

    if (files_only) {
        if (node->isFile()) { // Use isFile() which might trigger getNodeType()
            collected_nodes_.push_back(node);
        }
    } else { // Collect all node types
        collected_nodes_.push_back(node);
    }

    if (node->isDirectory()) {
        // Access children carefully. If they aren't populated, this iterator might be shallow.
        // For a deep iterator, ensure children are populated.
        // This iterator becomes "live" if getChildren might trigger parsing.
        // For simplicity, assume getChildren returns already parsed children.
        // User should call populateChildren(true) on root for a full deep iteration.
        std::vector<std::shared_ptr<PathNode>> children_copy = node->getChildren(); // Get const ref then copy if needed
        for (const auto& child : children_copy) {
             if (child->getDepth() < node->getDepth() + 100) // Safety break for deep recursion / symlink loops in iterator
                collectNodesRecursively(child, files_only);
        }
    }
}

PathNode::DescendantFileIterator& PathNode::DescendantFileIterator::operator++() {
    if (current_index_ < collected_nodes_.size()) {
        ++current_index_;
    }
    return *this;
}

std::shared_ptr<PathNode> PathNode::DescendantFileIterator::operator*() const {
    if (current_index_ < collected_nodes_.size()) {
        return collected_nodes_[current_index_];
    }
    // Or throw, or return a static null shared_ptr
    // Returning nullptr is common for end/invalid iterators but can lead to crashes if not checked.
    static std::shared_ptr<PathNode> null_node_ptr = nullptr;
    return null_node_ptr;
}

bool PathNode::DescendantFileIterator::operator==(const DescendantFileIterator& other) const {
    // Iterators are equal if they point to the same node in the same collection
    // or if both are "end" iterators (current_index_ at size) for their respective collections.
    // A simple check:
    bool this_is_end = current_index_ >= collected_nodes_.size();
    bool other_is_end = other.current_index_ >= other.collected_nodes_.size();

    if (this_is_end && other_is_end) return true;
    if (this_is_end != other_is_end) return false;

    // If both are not end, compare the actual pointers to the collected vectors and indices
    // This assumes iterators are only comparable if from the same "begin" call or both are "end"
    // For this simplified example, comparing current element might be enough if not comparing iterators from different collections.
    if (collected_nodes_.empty() && other.collected_nodes_.empty()) return true; // both generic end iterators
    
    // If they originate from the same collection (same root_node implicitly):
    return (&collected_nodes_ == &other.collected_nodes_) && (current_index_ == other.current_index_);
    // A more robust comparison might be needed if iterators can come from different "begin" calls on different subtrees.
}

bool PathNode::DescendantFileIterator::operator!=(const DescendantFileIterator& other) const {
    return !(*this == other);
}

PathNode::DescendantFileIterator PathNode::begin_files() {
    // For the iterator to work deeply, the tree needs to be populated.
    // Consider if begin_files should trigger a deep populate if not already done.
    // For now, it iterates what's currently in memory.
    // this->populateChildren(true); // <<<< If you want begin() to always be deep
    return DescendantFileIterator(shared_from_this(), true);
}

PathNode::DescendantFileIterator PathNode::end_files() {
    return DescendantFileIterator(); // Default constructor is the end iterator
}


// --- PathTree Implementation ---
PathTree::PathTree(const std::string& root_path_str, int parse_depth_limit)
    : PathNode(fs::absolute(root_path_str).lexically_normal().string(), /*parent*/ nullptr, /*tree_head*/ nullptr, /*depth*/ 0, parse_depth_limit)
{
    tree_head_ = std::static_pointer_cast<PathTree>(shared_from_this()); // Set self as head
    // Automatically populate the tree up to the depth limit upon construction
    // This can be a long operation.
    if(this->getNodeType() == NodeType::UNINITIALIZED) { // Update self first
        this->updateCachedInfoFromFilesystem();
    }
    if (this->isDirectory()) {
        this->populateChildren(true); // true for recursive population
    }
}

std::string PathTree::getFullPath() const { // Override from PathNode
    return name_;
}
} // namespace utils::system
#endif // __PATH_TREE_H__