using AbstractTrees

struct LineageTreeNode
    id::Int
    state::Vector{<:AbstractFloat}
    generation::Int
    children::Vector{LineageTreeNode}
    LineageTreeNode(id::Integer, state::Vector{<:AbstractFloat}, generation::Int, children::Vector{LineageTreeNode}=LineageTreeNode[]) = new(id, state, generation, children)
end

AbstractTrees.children(node::LineageTreeNode) = node.children
AbstractTrees.printnode(io::IO, node::LineageTreeNode) = print(io, "#", node.id, ", generation = ", node.generation)

struct LineageTree
    nodes::Dict{Int, LineageTreeNode}
    root::LineageTreeNode
end

_make_lintreenode((id, state, generation)::Tuple{<:Integer, <:Vector{<:AbstractFloat}, <:Integer}) = LineageTreeNode(id, state, generation)
_make_lintreenode((id, children)::Pair{<:Integer, <:Any}) = LineageTreeNode(id, _make_lintreenode.(children))

function LineageTree(x)
    root = _make_lintreenode(x)
    nodes = Dict{Int, LineageTreeNode}()
    for node in PreOrderDFS(root)
        haskey(nodes, node.id) && error("Duplicate node ID $(node.id)")
        nodes[node.id] = node
    end
    @info nodes, root
    return LineageTree(nodes, root)
end

function accum_state_probs(n::LineageTreeNode)
    n.state .= length(n.children) > 0 ? sum([accum_state_probs(c)/length(n.children) for c in n.children]) : n.state
end

function accum_state_num(n::LineageTreeNode)
    n.state .= length(n.children) > 0 ? sum([accum_state_num(c) for c in n.children]) : n.state
end
