using LinearAlgebra;
using Random;
using Distributed
using JLD2, FileIO, StatsBase, Dates
include("z2algebra.jl")


#-------------------------------------------------
#-------------------------------------------------
mutable struct StateDensityMatrix
    #save the stabs in column major fomat. each column is a Pauli string. first n rows represent the X part
    tab::Array{UInt8,2}
    phases::Array{UInt8}
    rank::Int
end
#-------------------------------------------------
#-------------------------------------------------
function productsign(a,b)
    
    #returns sign of a*b, when written in the standard form.
    #*0x02 is needed since we represent -1 as 0x02 (0x01 is imaginary i)
    
    n=length(a)÷2
    return (sum(a[n+1:end].*b[1:n])%0x02)*0x02
    
end
#-------------------------------------------------
#-------------------------------------------------
function commute(g::Array{UInt8,1},p::Array{UInt8,1})
    #whether g and p commute    
    n=length(g)÷2
    return sum(g[1:n].*p[n+1:2n]+g[n+1:2n].*p[1:n])%0x02
    
end
#-------------------------------------------------
#-------------------------------------------------
function commute(tab::Array{UInt8,2},g::Array{UInt8,1})
    #whether rows of tab commute with g.
    n=length(g)÷2
    return (transpose(tab[n+1:2n,:])*g[1:n].+transpose(tab[1:n,:])*g[n+1:2n]).%0x02
    
end
#-------------------------------------------------
#-------------------------------------------------
function addCols(state::StateDensityMatrix,i,j)
    
    #replaces g_j with gi*g_j in the tableau. takes care of phases
    
    n=size(state.tab,1)÷2
    state.phases[j]=(state.phases[j]+state.phases[i]+productsign(state.tab[:,j],state.tab[:,i]))%0x04
    state.tab[:,j]=state.tab[:,j].⊻state.tab[:,i]
    return
    
end
#-------------------------------------------------
#-------------------------------------------------
function swapCols(state::StateDensityMatrix,i,j)
    
    #swaps stabilizer i and j. takes care of phases.
    
    temp_col=state.tab[:,i]
    temp_phase=state.phases[i]
    
    state.tab[:,i]=state.tab[:,j]
    state.phases[i]=state.phases[j]
    
    state.tab[:,j]=temp_col
    state.phases[j]=temp_phase
    
    return
end
#-------------------------------------------------
#-------------------------------------------------
function updateDestabilizers(state::StateDensityMatrix,i)
#     this function assumes i<=n
#     it also assumes that state.tab has a valid form except for some js with j>n and j!=i+n
#     ignores phases becasue the phase of destabilizers is irrelavant.
    n=size(state.tab,1)÷2
    
    commuteSign=commute(state.tab[:,n+1:2n],state.tab[:,i])
    commuteSign[i]=0x00
            
    noncommuting=findall(x->x==0x01,commuteSign).+n
    for j in noncommuting
        addCols(state,i+n,j)
    end
    
    return
end
#-------------------------------------------------
#-------------------------------------------------
function insertStabilizer(state::StateDensityMatrix,g,gphase,i)
#     moves the i'th stabilizer to the i'th destabilizer position and puts g as the i'th stabilizzer
    n=size(state.tab,1)÷2
    state.tab[:,i+n]=state.tab[:,i]
    state.tab[:,i]=g
    state.phases[i]=gphase
end
#-------------------------------------------------
#-------------------------------------------------
function measure(state::StateDensityMatrix,g,gphase=0x00,verbose=false)
    verbose && printstyled("\n measuring: $g\n";color=:blue)
    verbose && println("state has originally tableau:")
    verbose && display(state.tab)
    verbose && println("with rank:$(state.rank) and phases:$(state.phases)")
    
    n=size(state.tab,1)÷2
    r=state.rank
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],g))

    if length(noncommuting)==0
        verbose && println("no noncommuting in-group stabilizer")
        #check whether g is in the stabilizer group
        
        noncommuting_S=findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
        if length(noncommuting_S)!=0
            verbose && println("a non-commuting out-of-group stabilizer has been found:\n $(state.tab[:,noncommuting_S[1]])")

            #a non-trivial operator has been measured. 
            
            pivotIdx=noncommuting_S[1]
            
            #after this loop we have a valid table with a single stabilizer that anti-commutes with g
            for j in noncommuting_S[2:end]
                addCols(state,pivotIdx,j)
#                 the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#                 addCols(state,j+n,pivotIdx+n)     
            end

            outcome=rand([0x00,0x02])
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
                        
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group stabilizer")
        
        noncommuting_D=findall(x->x==0x01,commute(state.tab[:,r+1+n:2n],g)).+(r+n)
        if length(noncommuting_D)!=0
            verbose && println("a non-commuting out-of-group destabilizer has been found:\n $(state.tab[:,noncommuting_D[1]])")
#             a non-trivial operator has been measured. 
#             we swap the stabilizer and destabilizer for the first non-commuting destabilizer
#             By doing so, we arrive at a valid tableau and a single stabilizer that anti-commutes with g
            swapCols(state,noncommuting_D[1],noncommuting_D[1]-n)
            pivotIdx=noncommuting_D[1]-n
            
            outcome=rand([0x00,0x02])
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
            
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)            
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group destabilizer.\n g is in the stabilizer group")

        
#         +-g is in the stabilizer group, so the measurement outcome is deterministic and rank remains the same
        noncommuting_DS=findall(x->x==0x01,commute(state.tab[:,1+n:r+n],g))
        if length(noncommuting_DS)==0
            if sum(g)==0
                return 1
            else
                printstyled("ERROR: SOMETHING IS VERY WRONG: g commutes with everything!\n";color = :red)
                println("g:")
                display(g)
                println("tab:")
                display(state.tab)
                println("rank:$(state.rank)")
                return
            end
        end
        ph=state.phases[noncommuting_DS[1]]
        op=state.tab[:,noncommuting_DS[1]]
        
        for i in noncommuting_DS[2:end]
            ph=ph+state.phases[i]+productsign(op,state.tab[:,i])
            op=op.⊻state.tab[:,i]
        end
        ph=ph%0x04
        
        if g!=op
            
            printstyled("ERROR: SOMETHING IS VERY WRONG:\n $g \n is not equal to\n $op\n";color = :red)
            println("g:")
            display(g)
            println("tab:")
            display(state.tab)
            println("rank:$(state.rank)")
        end
        if gphase==ph
            return 0x00
        elseif gphase==(ph+0x02)%0x04
            return 0x02
        else
            printstyled("ERROR: SOMETHING IS VERY WRONG. MAYBE YOU ARE MEASURING AN ANTI-HERMITIONA OPERATOR\n";color = :red)
            return 
        end
            
    else
        verbose && println("g anti-commutes with a stabilizer: $(state.tab[:,noncommuting[1]])")
        #the measurement is not compatible with the stabilizer group. the outcome is random
        pivotIdx=noncommuting[1]
        for j in noncommuting[2:end]
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end
        
        #update the out-of-group stabilizers, so tehy commute with the newly added stabilizer
        for j in findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end

        outcome=rand([0x00,0x02])
        insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
        updateDestabilizers(state,pivotIdx)
        return outcome
    end
end
#-------------------------------------------------
#-------------------------------------------------

function forceMeasure(state,g,gphase=0x00,verbose=false)
#     the only deference between this function and measure is:
#     1) the phase of g will not be randomized when g is not in the stabilizer group.
#     2) if -gphase*g is in the stabilizer group, it kills the state (sets it to zero) 


    
    verbose && printstyled("\n projecting on: $g\n with phase:$gphase ";color=:blue)
    verbose && println("state's tableau:")
    verbose && display(state.tab)
    verbose && println("rank:$(state.rank), phases:$(state.phases)")

    
    n=size(state.tab,1)÷2
    r=state.rank
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],g))

    if length(noncommuting)==0
        verbose && println("no noncommuting in-group stabilizer")
        #check whether g is in the stabilizer group
        
        noncommuting_S=findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
        if length(noncommuting_S)!=0
            verbose && println("a non-commuting out-of-group stabilizer has been found:\n $(state.tab[:,noncommuting_S[1]])")

            #a non-trivial operator has been measured. 
            
            pivotIdx=noncommuting_S[1]
            
            #after this loop we have a valid table with a single stabilizer that anti-commutes with g
            for j in noncommuting_S[2:end]
                addCols(state,pivotIdx,j)
#                 the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#                 addCols(state,j+n,pivotIdx+n)     
            end

#             since it is forced measurement
#             outcome=rand([0x00,0x02])
            outcome=0x00
            
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
                        
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group stabilizer")
        
        noncommuting_D=findall(x->x==0x01,commute(state.tab[:,r+1+n:2n],g)).+(r+n)
        if length(noncommuting_D)!=0
            verbose && println("a non-commuting out-of-group destabilizer has been found:\n $(state.tab[:,noncommuting_D[1]])")
#             a non-trivial operator has been measured. 
#             we swap the stabilizer and destabilizer for the first non-commuting destabilizer
#             By doing so, we arrive at a valid tableau and a single stabilizer that anti-commutes with g
            swapCols(state,noncommuting_D[1],noncommuting_D[1]-n)
            pivotIdx=noncommuting_D[1]-n
            
#             since it is forced measurement
#             outcome=rand([0x00,0x02])
            outcome=0x00
            
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
            
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)            
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group destabilizer.\n g is in the stabilizer group")

        
#         +-g is in the stabilizer group, so the measurement outcome is deterministic and rank remains the same
        noncommuting_DS=findall(x->x==0x01,commute(state.tab[:,1+n:r+n],g))
        if length(noncommuting_DS)==0
            if sum(g)==0
                return 1
            else
                printstyled("ERROR: SOMETHING IS VERY WRONG: g commutes with everything!\n";color = :red)
                println("g:")
                display(g)
                println("tab:")
                display(state.tab)
                println("rank:$(state.rank)")
                return
            end
        end
        ph=state.phases[noncommuting_DS[1]]
        op=state.tab[:,noncommuting_DS[1]]
        
        for i in noncommuting_DS[2:end]
            ph=ph+state.phases[i]+productsign(op,state.tab[:,i])
            op=op.⊻state.tab[:,i]
        end
        ph=ph%0x04
        
        if g!=op
            
            printstyled("ERROR: SOMETHING IS VERY WRONG:\n $g \n is not equal to\n $op\n";color = :red)
            println("g:")
            display(g)
            println("tab:")
            display(state.tab)
            println("rank:$(state.rank)")
        end
        if gphase==ph
            return 0x00
        elseif gphase==(ph+0x02)%0x04
                printstyled("WARNING: THE STATE HAD NO AMPLITITUDE IN THE FORCED SUBSPACE. NOTHING WILL BE DONE\n";color = :red)
            return 0x00
        else
            printstyled("ERROR: SOMETHING IS VERY WRONG. MAYBE YOU ARE MEASURING AN ANTI-HERMITIONA OPERATOR\n";color = :red)
            return 
        end
            
    else
        verbose && println("g anti-commutes with a stabilizer: $(state.tab[:,noncommuting[1]])")
        #the measurement is not compatible with the stabilizer group. the outcome is random
        pivotIdx=noncommuting[1]
        for j in noncommuting[2:end]
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end
        
        #update the out-of-group stabilizers, so tehy commute with the newly added stabilizer
        for j in findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end

#         since it is forced measurement
#         outcome=rand([0x00,0x02])
        outcome=0x00
        
        insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
        updateDestabilizers(state,pivotIdx)
        return outcome
    end
end

#-------------------------------------------------
#-------------------------------------------------       
function totallyMixed(n)
    state=StateDensityMatrix(Matrix{UInt8}(I,2n,2n),zeros(UInt8,2n),0)
    return state
end
#-------------------------------------------------
#-------------------------------------------------          
function xProductState(n)
    state=StateDensityMatrix(Matrix{UInt8}(I,2n,2n),zeros(UInt8,2n),n)
    return state
end
#-------------------------------------------------
#-------------------------------------------------
function zProductState(n)
    state=StateDensityMatrix(zeros(2n,2n),zeros(UInt8,2n),n)
    state.tab[n+1:2n,1:n]=Matrix{UInt8}(I,n,n)
    state.tab[1:n,1+n:2n]=Matrix{UInt8}(I,n,n)
    return state
end
#-------------------------------------------------
#-------------------------------------------------

# function applyUnitary(state,u,uphase,indices)
# #     applies unitary gate "u" to qubits with indexs given by "indices"
# #     u is assumed to be given in the following form. say length indices =k,
# #     i.e. u acts on k qubits. Then u is a 2k x 2k bit matrix where the first
# #     k columns are images of X1,...,Xk in the form of Pauli strings (bit 
# #     strings of length 2k) and the rest of the columns (k+1 to 2k) are 
# #     the images of Z1,...,Zk.
    
#     n=size(state.tab,1)÷2

#     for i=1:size(stab,1)
#         js=findall(isequal(0x01),@view stab[i,[indices;indices.+n]])
#         if length(js)==0 continue end

#         img=u[:,js[1]]
#         sign=usign[js[1]]

#         for j in js[2:end]
#             sign⊻=usign[j]⊻ productsign(img,u[:,j])
#             img.⊻=u[:,j]
#         end

#         stabsign[i]⊻=sign
#         stab[i,[indices;indices.+n]]=img
#     end
#     return
# end

#-------------------------------------------------
#-------------------------------------------------

function decohereX(state,i)
    n=size(state.tab,1)÷2
    r=state.rank
    Xi=zeros(UInt8,2n)
    Xi[i]=0x01
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],Xi))
    if length(noncommuting)==0
        return 0x00
    else
        pivotIdx=noncommuting[1]
#         after this for loop, we would have a valid stabilizer tableau where
#         where only one stabilizer anticommutes with X_i
        
        for j in noncommuting[2:end]
            addCols(state,pivotIdx,j)
#             this is needed to keep the stabilizer/destabilizer valid
            addCols(state,j+n,pivotIdx+n)  
        end
        
        swapCols(state,pivotIdx,r)
        swapCols(state,pivotIdx+n,r+n)
        
        state.rank-=1
        return 0x01
    end
end

#-------------------------------------------------
#-------------------------------------------------

function expval(state,g,gphase)
    verbose && printstyled("\n calculating expectation value of: $g\n";color=:blue)
    verbose && println("state has the following tableau:")
    verbose && display(state.tab)
    verbose && println("with rank:$(state.rank) and phases:$(state.phases)")
    
    n=size(state.tab,1)÷2
    r=state.rank
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],g))

    if length(noncommuting)==0
        verbose && println("no noncommuting in-group stabilizer")
        #check whether g is in the stabilizer group
        
        noncommuting_S=findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
        if length(noncommuting_S)!=0
            verbose && println("a non-commuting out-of-group stabilizer has been found: g is not in the stabilizer group")
            return 0
        end
        verbose && println("no noncommuting out-of-group stabilizer has been found")
        
        noncommuting_D=findall(x->x==0x01,commute(state.tab[:,r+1+n:2n],g)).+(r+n)
        if length(noncommuting_D)!=0
            verbose && println("a non-commuting out-of-group destabilizer has been found: g is not in the stabilizer group")
            return 0
        end
        verbose && println("no noncommuting out-of-group destabilizer.\n g is in the stabilizer group")

        
#         +-g is in the stabilizer group, so the measurement outcome is deterministic and rank remains the same
        noncommuting_DS=findall(x->x==0x01,commute(state.tab[:,1+n:r+n],g))
        if length(noncommuting_DS)==0
            if sum(g)==0
                return 1
            else
                printstyled("ERROR: SOMETHING IS VERY WRONG: g commutes with everything!\n";color = :red)
                println("g:")
                display(g)
                println("tab:")
                display(state.tab)
                println("rank:$(state.rank)")
                return
            end
        end
        ph=state.phases[noncommuting_DS[1]]
        op=state.tab[:,noncommuting_DS[1]]
        
        for i in noncommuting_DS[2:end]
            ph=ph+state.phases[i]+productsign(op,state.tab[:,i])
            op=op.⊻state.tab[:,i]
        end
        ph=ph%0x04
        
        if g!=op
            
            printstyled("ERROR: SOMETHING IS VERY WRONG:\n $g \n is not equal to\n $op\n";color = :red)
            println("g:")
            display(g)
            println("tab:")
            display(state.tab)
            println("rank:$(state.rank)")
        end
        if gphase==ph
            return 0x00
        elseif gphase==(ph+0x02)%0x04
            return 0x02
        else
            printstyled("ERROR: SOMETHING IS VERY WRONG. MAYBE g IS NOT HERMITIONA\n";color = :red)
            return 
        end
            
    else
        verbose && println("g anti-commutes with a stabilizer")
        return 0
    end
